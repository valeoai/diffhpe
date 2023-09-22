import torch
import torch.nn as nn
from diffhpe.architectures import LiftCTGCN
from diffhpe.architectures import MixSTE
from diffhpe.architectures import MlpCoords, MlpJoints, MlpSeq


class LiftingSupervisedDebugModel(nn.Module):
    n_input_coords = 2

    def __init__(self, config, device, skeleton, seq_len):
        super().__init__()
        self.device = device
        self.seq_len = seq_len
        self.skeleton = skeleton
        self.debug = config.debug

        config_diff = config.diffusion

        # Whether to predict the whole 3D pose or only the depth
        if config_diff.cond_mix_mode == "z_only":
            self.n_output_coords = 1
        elif config_diff.cond_mix_mode == "xy_only":
            self.n_output_coords = 2
        else:
            self.n_output_coords = 3

        self.target_mode = config_diff.cond_mix_mode
        self.input_mode = config_diff.conditioning

        if config_diff.arch == "gcn":
            self.model = LiftCTGCN(
                config_diff,
                seq_len=seq_len,
                skeleton=skeleton,
                n_input_coords=self.n_input_coords,
                n_output_coords=self.n_output_coords,
                debug_disable_diffusion=True,
            )
        elif config_diff.arch == "mixste":
            self.model = MixSTE(
                num_frame=seq_len,
                num_joints=skeleton.num_joints(),
                in_chans=self.n_input_coords,
                out_dim=self.n_output_coords,
                embed_dim=config_diff.channels,
                depth=config_diff.layers,
                num_heads=config_diff.nheads,
                drop_path_rate=config_diff.drop_path_rate,
            )
        elif config_diff.arch == "mlp_coord":
            self.model = MlpCoords(
                in_features=self.n_input_coords,
                hidden_features=config_diff.channels,
                out_features=self.n_output_coords,
                n_layers=config_diff.layers,
                drop=config_diff.p_dropout,
            )
        elif config_diff.arch == "mlp_joints":
            self.model = MlpJoints(
                in_features=self.n_input_coords * skeleton.num_joints(),
                hidden_features=config_diff.channels,
                out_features=self.n_output_coords * skeleton.num_joints(),
                n_layers=config_diff.layers,
                drop=config_diff.p_dropout,
            )
        elif config_diff.arch == "mlp_seq":
            n_joints_x_seq = skeleton.num_joints() * seq_len
            self.model = MlpSeq(
                in_features=self.n_input_coords * n_joints_x_seq,
                hidden_features=config_diff.channels,
                out_features=self.n_output_coords * n_joints_x_seq,
                n_layers=config_diff.layers,
                drop=config_diff.p_dropout,
            )
        else:
            raise ValueError(
                "Architecture param could not be recognized: "
                f"{config_diff['arch']}. Only gcn is implemented for lifting."
            )

    def calc_loss_valid(self, data_2d, data_3d, is_train):
        return self.calc_loss(data_2d, data_3d, is_train)

    def calc_loss(self, data_2d, data_3d, is_train, set_t=None):
        if self.input_mode == "gt_xy":
            data_2d = data_3d[:, :2, :, :]

        B, _, J, L = data_3d.shape
        if self.target_mode == "z_only":
            data_3d = data_3d[:, 2, :, :].view(B, 1, J, L)
        elif self.target_mode == "xy_only":
            data_3d = data_3d[:, :2, :, :]

        predicted = self.model(data_2d, torch.zeros(1, device=self.device))
        residual = data_3d - predicted
        loss = (residual**2).mean()
        return loss

    def impute(self, data_2d, n_samples):
        B, _, J, L = data_2d.shape
        target_3d = self.model(data_2d, torch.zeros(1, device=self.device))
        # (B,3,J,L)

        assert target_3d.shape[1] == self.n_output_coords

        imputed_samples = (
            target_3d.detach()
            .reshape(B, 1, self.n_output_coords, J, L)
            .repeat((1, n_samples, 1, 1, 1))
        )
        return imputed_samples

    def forward(self, batch, is_train=1):
        (
            data_3d,
            data_2d,
        ) = self.process_data(batch, get_3d=True)
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        return loss_func(data_2d, data_3d, is_train)

    def evaluate(self, batch, n_samples):
        get_3d = (
            self.input_mode == "gt_xy"
            or self.target_mode == "z_only"
            or self.target_mode == "xy_only"
        )

        data_3d, data_2d = self.process_data(batch, get_3d=get_3d)
        if self.input_mode == "gt_xy":
            data_2d = data_3d[:, :2, :, :]
        with torch.no_grad():
            samples = self.impute(data_2d, n_samples)
            if self.target_mode == "z_only":
                B, S, _, J, L = samples.shape
                z_pred = samples.clone()
                samples = torch.empty(B, S, 3, J, L)
                # Use input x,y and output z
                samples[:, :, 2:, :, :] = z_pred
                samples[:, :, :2, :, :] = data_3d[:, :2, :, :].unsqueeze(1)
            elif self.target_mode == "xy_only":
                B, S, _, J, L = samples.shape
                xy_pred = samples.clone()
                samples = torch.empty(B, S, 3, J, L)
                # Use input x,y and output z
                samples[:, :, :2, :, :] = xy_pred
                samples[:, :, 2, :, :] = data_3d[:, 2, :, :].unsqueeze(1)
        return samples

    def process_data(self, batch, get_3d):
        pose_2d = batch["pose_2d"].to(self.device).float()  # (B, L, J, 2)
        # TODO: Implement masks if I plan on plugging onto forecaster
        # mask = batch["mask"].to(self.device).float()  # (B, L, J, 3)

        pose_2d = pose_2d.permute(0, 3, 2, 1)  # (B, 2, J, L)
        # mask = mask.permute(0, 3, 2, 1)  # (B, 3, J, L)

        pose_3d = None
        if get_3d:
            pose_3d = batch["pose_3d"].to(self.device).float()  # (B, L, J, 3)
            pose_3d = pose_3d.permute(0, 3, 2, 1)  # (B, 3, J, L)

        return (
            pose_3d,
            pose_2d,
        )
