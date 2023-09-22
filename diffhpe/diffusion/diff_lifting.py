import numpy as np
import torch
import torch.nn as nn

from diffhpe.architectures import LiftCTGCN
from diffhpe.architectures import DiffMixSTE
from diffhpe.diffusion.conditioners import (
    MixSTECond,
    PTMixSTECond,
    UVCond,
    NoConditioning,
)
from diffhpe.diffusion.conditioners import apply_guidance
from diffhpe.metrics import sagittal_symmetry, segments_time_consistency
from .utils import compute_noise_scheduling


class LiftingDiffusionModel(nn.Module):
    def __init__(self, config, device, skeleton, seq_len):
        super().__init__()
        self.device = device
        self.seq_len = seq_len
        self.skeleton = skeleton

        self.debug = config.debug

        config_diff = config.diffusion
        self.p_uncond = config_diff.p_uncond
        self.guidance = config_diff.guidance
        self.guidance_before_cond = config_diff.guidance_before_cond
        self.rand_guidance_rate = config_diff.rand_guidance_rate
        if self.rand_guidance_rate < 1.0:
            assert self.guidance_before_cond, (
                "Random guidance is only implemented when guidance if carried"
                "before conditioning for now. Set guidance_before_cond=True."
            )

        # Define conditioning model, which can either leave
        # data_2d unchanged (as we do now) or modify it (i.e. passing through
        # another model, like in DiffPose)
        # The concatenation can also be part of the model (which would take
        # noisy_data and data_2d as input), so that we can easily change to
        # conditioning through addition if we want
        # Conditioning models I'd like to test:
        #  - simple projector to high dim, followed by addition or concat (as
        #    in unpublished DiffPose)
        #  - pre-trained MixSTE without its head, as done in published DiffPose
        #  - simple (x,y,0) predictions derived determinitically from input_2d
        #    using camera params or with learned intrinsic params
        self._extra_input = None

        if config_diff.conditioning == "raw":
            self.conditioning = UVCond(
                mix_mode=config_diff.cond_mix_mode,
            )
        elif config_diff.conditioning == "uncond":
            self.conditioning = NoConditioning()
        elif config_diff.conditioning == "pt_mixste":
            self.conditioning = PTMixSTECond(
                mix_mode=config_diff.cond_mix_mode,
                encode_3d=config_diff.cond_encode_3d,
                keep_pt_head=config_diff.cond_keep_pt_head,
                checkpoint_path=config_diff.cond_ckpt,
                num_frame=seq_len,
                num_joints=skeleton.num_joints(),
                embed_dim=config_diff.cond_channels,
                depth=config_diff.cond_layers,
                num_heads=config_diff.cond_nheads,
            )
        elif config_diff.conditioning == "untrained_mixste":
            self.conditioning = MixSTECond(
                mix_mode=config_diff.cond_mix_mode,
                encode_3d=config_diff.cond_encode_3d,
                num_frame=seq_len,
                num_joints=skeleton.num_joints(),
                embed_dim=config_diff.cond_channels,
                depth=config_diff.cond_layers,
                num_heads=config_diff.cond_nheads,
                drop_path_rate=config_diff.conf_drop_path_rate,
            )
        else:
            raise ValueError(
                "Invalid value for conditioning param."
                "Possible values are raw, pt_mixste, untrained_mixste, gt_xy, "
                "known_pinhole, learned_pinhole."
            )
        self.conditioning.to(self.device)

        # Whether to predict the whole 3D pose or only the depth
        self.n_output_coords = (
            1 if self.conditioning.mix_mode == "z_only" else 3
        )

        if config_diff.arch == "gcn":
            self.diffmodel = LiftCTGCN(
                config_diff,
                seq_len=seq_len,
                skeleton=skeleton,
                n_input_coords=self.conditioning.out_dim,
                n_output_coords=self.n_output_coords,
            )
        elif config_diff.arch == "mixste":
            self.diffmodel = DiffMixSTE(
                num_frame=seq_len,
                num_joints=skeleton.num_joints(),
                in_chans=self.conditioning.out_dim,
                out_dim=self.n_output_coords,
                embed_dim=config_diff.channels,
                depth=config_diff.layers,
                num_heads=config_diff.nheads,
                drop_path_rate=config_diff.drop_path_rate,
            )
        else:
            raise ValueError(
                "Architecture param could not be recognized: "
                f"{config_diff['arch']}. Only gcn is implemented for lifting."
            )

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        (
            self.beta,
            self.alpha,
            self.alpha_hat,
            self.sigma,
        ) = compute_noise_scheduling(
            schedule=config_diff.schedule,
            beta_start=config_diff.beta_start,
            beta_end=config_diff.beta_end,
            num_steps=config_diff.num_steps,
        )

        self.alpha_torch = (
            torch.tensor(self.alpha)
            .float()
            .to(self.device)
            .unsqueeze(1)
            .unsqueeze(1)
        )

        self.skeleton_time_regularization = (
            config_diff.skeleton_time_regularization
        )
        self.skeleton_side_regularization = (
            config_diff.skeleton_side_regularization
        )

        # Load some covariance and mean to converge toward anisotropic Gaussian
        self.anisotropic = config_diff.anisotropic
        if self.anisotropic:
            err_var = (
                torch.from_numpy(
                    np.load(config_diff.err_var_path),
                )
                .float()
                .to(self.device)
            )
            self.err_sqrt_cov = torch.diag(torch.sqrt(err_var).flatten())
            err_mean = (
                torch.from_numpy(
                    np.load(config_diff.err_ave_path),
                )
                .float()
                .to(self.device)
            )
            self.err_mean = err_mean.flatten().unsqueeze(-1)

    def calc_loss_valid(self, data_2d, data_3d, is_train, cam=None):
        loss_sum = 0
        for t in range(0, self.num_steps, 10):  # calculate loss for fixed t
            loss = self.calc_loss(data_2d, data_3d, is_train, set_t=t, cam=cam)
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(self, data_2d, data_3d, is_train, set_t=-1, cam=None):
        B, _, J, L = data_3d.shape
        if self.debug.overfit_last_diffusion_step:
            DIFFUSION_STEP_TO_OVERFIT = 1
            t = (
                (torch.ones(B) * DIFFUSION_STEP_TO_OVERFIT)
                .long()
                .to(self.device)
            )
        else:
            if is_train == 1:
                t = torch.randint(0, self.num_steps, [B]).to(self.device)
            else:
                # for validation
                t = (torch.ones(B) * set_t).long().to(self.device)
        current_alpha = (
            self.alpha_torch[t].to(self.device).unsqueeze(-1)
        )  # (B,1,1,1)

        noise = torch.randn_like(data_3d).to(self.device)  # (B,3,J,L)
        if self.anisotropic:
            noise = (
                torch.mm(
                    self.err_sqrt_cov,
                    noise.permute(2, 1, 0, 3).reshape(J * 3, B * L),
                )
                + self.err_mean
            )
            noise = noise.reshape(J, 3, B, L).permute(2, 1, 0, 3)

        noisy_data = (current_alpha**0.5) * data_3d + (
            1.0 - current_alpha
        ) ** 0.5 * noise  # (B,3,J,L)

        # Drop x,y coordinates in diffused data
        if self.conditioning.mix_mode == "z_only":
            noise = noise[:, 2, :, :].view(B, 1, J, L)
            noisy_data = noisy_data[:, 2, :, :].view(B, 1, J, L)

        total_input = self.apply_conditioning(
            noisy_data,
            data_2d,
            cam,
            data_3d,
            p_uncond=self.p_uncond,
        )

        predicted = self.diffmodel(total_input, t)  # (B,3,J,L)
        residual = noise - predicted
        loss = (residual**2).mean()

        if (
            self.skeleton_time_regularization > 0.0
            or self.skeleton_side_regularization > 0.0
        ):
            # The original data_3d can be recovered by:
            # data_3d = (
            # noisy_data - (1.0 - current_alpha) ** 0.5 * noise
            # ) / (current_alpha**0.5)
            # we will attempt to reconstruct it from the predicted noise
            data_3d = (
                noisy_data - (1.0 - current_alpha) ** 0.5 * predicted
            ) / (current_alpha**0.5)

            bones_time_variance = segments_time_consistency(
                data_3d,
                self.skeleton,
            )
            bones_side_difference = sagittal_symmetry(
                data_3d,
                self.skeleton,
            )
            loss = (
                (
                    1.0
                    - self.skeleton_time_regularization
                    - self.skeleton_side_regularization
                )
                * loss
                + self.skeleton_time_regularization * bones_time_variance
                + self.skeleton_side_regularization * bones_side_difference
            )

        return loss

    def apply_conditioning(
        self, target_3d, data_2d, cam, data_3d, p_uncond=0.0
    ):
        if self.guidance_before_cond:
            # Classifier-free guidance before conditioners
            data_2d = apply_guidance(
                data_2d, p_uncond, self.rand_guidance_rate
            )
            p_uncond = 0.0  # disable guidance after conditioning

        if self._extra_input is None:
            diff_input = self.conditioning(
                target_3d,
                data_2d,
                p_uncond=p_uncond,
            )
        elif self._extra_input == "cam":
            diff_input = self.conditioning(
                target_3d,
                data_2d,
                cam,
                p_uncond=p_uncond,
            )
        elif self._extra_input == "gt":
            diff_input = self.conditioning(
                target_3d,
                data_3d,
                p_uncond=p_uncond,
            )
        else:
            raise ValueError(
                "_extra_input can only take values None, 'cam' and "
                f"'gt'. Got {self._extra_input}."
            )
        return diff_input

    def impute(
        self,
        data_2d,
        n_samples,
        cam=None,
        data_3d=None,
        starting_pose=None,
        n_steps=None,
    ):
        B, D, J, L = data_2d.shape
        assert D == 2, f"Expected input dimension 2, got {D}."

        if n_steps is None:
            n_steps = self.num_steps

        imputed_samples = torch.zeros(B, n_samples, 3, J, L).to(self.device)

        for i in range(n_samples):
            if starting_pose is None:
                target_3d = torch.randn(
                    (B, self.n_output_coords, J, L),
                    device=data_2d.device,
                )

                if self.anisotropic:
                    target_3d = (
                        torch.mm(
                            self.err_sqrt_cov,
                            target_3d.permute(2, 1, 0, 3).reshape(
                                J * 3, B * L
                            ),
                        )
                        + self.err_mean
                    )
                    target_3d = target_3d.reshape(J, 3, B, L).permute(
                        2, 1, 0, 3
                    )
            else:
                assert isinstance(starting_pose, torch.Tensor), (
                    "stating_pose should be a torch tensor. "
                    f"Got {type(starting_pose)}."
                )
                assert starting_pose.shape == (B, self.n_output_coords, J, L)
                target_3d = starting_pose.to(data_2d.device)

            for t in range(n_steps - 1, -1, -1):
                diff_input = self.apply_conditioning(
                    target_3d,
                    data_2d,
                    cam,
                    data_3d,
                    p_uncond=0.0,
                )

                # Conditional noise prediction
                predicted_noise = self.diffmodel(
                    diff_input, torch.tensor([t]).to(self.device)
                )  # (B,3,J,L)

                # Classifier-free guidance
                if self.p_uncond > 0 and self.guidance > 0.0:
                    # Unconditional input
                    diff_input_uncond = self.apply_conditioning(
                        target_3d,
                        data_2d,
                        cam,
                        data_3d,
                        p_uncond=1.0,
                    )
                    predicted_noise *= 1 + self.guidance
                    predicted_noise -= self.guidance * self.diffmodel(
                        diff_input_uncond, torch.tensor([t]).to(self.device)
                    )  # (B,3,J,L)

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_mean = coeff1 * (target_3d - coeff2 * predicted_noise)

                if t > 0:
                    noise = torch.randn_like(current_mean)
                    target_3d = current_mean + self.sigma[t - 1] * noise

            if self.conditioning.mix_mode != "z_only":
                imputed_samples[:, i] = current_mean.detach()
            else:
                # Use input x,y and output z
                imputed_samples[:, i, 2:, :, :] = current_mean.detach()
                imputed_samples[:, i, :2, :, :] = diff_input[:, 1:].detach()
        return imputed_samples

    def forward(self, batch, is_train=1):
        (
            data_3d,
            data_2d,
            cam,
        ) = self.process_data(batch, get_3d=True)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(data_2d, data_3d, is_train, cam=cam)

    def evaluate(self, batch, n_samples, starting_pose=None, n_steps=None):
        get_3d = self._extra_input == "gt"
        data_3d, data_2d, cam = self.process_data(batch, get_3d=get_3d)

        with torch.no_grad():
            samples = self.impute(
                data_2d,
                n_samples,
                cam=cam,
                data_3d=data_3d,
                starting_pose=starting_pose,
                n_steps=n_steps,
            )
        return samples

    def process_data(self, batch, get_3d):
        pose_2d = batch["pose_2d"].to(self.device).float()  # (B, L, J, 2)
        # TODO: Implement masks if I plan on plugging onto forecaster
        # mask = batch["mask"].to(self.device).float()  # (B, L, J, 3)

        pose_2d = pose_2d.permute(0, 3, 2, 1)  # (B, 2, J, L)
        # mask = mask.permute(0, 3, 2, 1)  # (B, 3, J, L)

        cam = batch["cam"].to(self.device).float()  # (B, L, 9)

        pose_3d = None
        if get_3d:
            pose_3d = batch["pose_3d"].to(self.device).float()  # (B, L, J, 3)
            pose_3d = pose_3d.permute(0, 3, 2, 1)  # (B, 3, J, L)

        return (
            pose_3d,
            pose_2d,
            cam,
        )
