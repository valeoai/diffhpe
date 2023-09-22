from __future__ import absolute_import, division, print_function

import os
import pickle
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from diffhpe.data.lifting.camera import camera_to_world, image_coordinates
from diffhpe.data.lifting.generators import PoseSequenceGenerator
from diffhpe.data.lifting.h36m_lifting import Human36mDataset
from diffhpe.data.lifting.utils import create_2d_data, read_3d_data
from diffhpe.diffusion.diff_lifting import LiftingDiffusionModel
from main_h36m_lifting import evaluate
from diffhpe.supervised.lifting import LiftingSupervisedDebugModel
from useful_scripts.visualization import render_animation


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("==> Using settings:")
    print(OmegaConf.to_yaml(cfg))

    orig_cwd = Path(get_original_cwd())
    figures_dir = orig_cwd / "figures"
    figures_data_dir = orig_cwd / "figures-data"

    # Load dataset
    data_dir = Path(cfg.data.data_dir)
    preproc_dataset_path = (
        data_dir / f"preproc_data_3d_{cfg.data.dataset}_{cfg.data.joints}.pkl"
    )

    if preproc_dataset_path.exists():
        print("==> Loading preprocessed dataset...")
        with open(preproc_dataset_path, "rb") as f:
            dataset = pickle.load(f)
    else:
        print("==> Loading raw dataset...")
        dataset_path = data_dir / f"data_3d_{cfg.data.dataset}.npz"

        dataset = Human36mDataset(dataset_path, n_joints=cfg.data.joints)

        print("==> Preparing data...")
        dataset = read_3d_data(dataset)

        print("==> Caching data...")
        with open(preproc_dataset_path, "wb") as f:
            pickle.dump(dataset, f)

    # And 2D data
    inputs_path = data_dir / (
        f"data_2d_{cfg.data.dataset}_{cfg.data.keypoints}.npz"
    )
    keypoints = create_2d_data(inputs_path, dataset)

    cudnn.benchmark = True
    device = torch.device("cuda")

    # Create model
    print("==> Creating model...")

    if cfg.diffusion.disable_diffusion_and_supervise:
        print(
            "DEBUG MODE: disabling diffusion mechanism and using supervised "
            "learning!"
        )
        if cfg.debug.overfit_last_diffusion_step:
            print(
                "WARNING: overfit_last_diffusion_step ignored when using "
                "supervised learning"
            )
        ModelClass = LiftingSupervisedDebugModel
    else:
        if cfg.debug.overfit_last_diffusion_step:
            print("DEBUG MODE: overfitting last diffusion step!")
        ModelClass = LiftingDiffusionModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: %s" % device)

    model_pos = ModelClass(
        cfg,
        device,
        skeleton=dataset.skeleton(),
        seq_len=cfg.data.seq_len,
    )

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_pos = nn.DataParallel(model_pos)

    model_pos.to(device)

    print(
        "==> Total parameters: {:.2f}M".format(
            sum(p.numel() for p in model_pos.parameters()) / 1000000.0
        )
    )

    # Resume from a checkpoint
    cwd = Path(os.getcwd())
    model_l_path = cwd / cfg.eval.model_l
    if model_l_path.is_dir():
        print("==> Loading checkpoint '{}'".format(model_l_path))
        model_pos.load_state_dict(torch.load(f"{model_l_path}/model.pth"))
    else:
        raise RuntimeError(
            "==> No checkpoint found at '{}'".format(model_l_path)
        )

    print("==> Rendering...")

    poses_2d_subj = {
        k.lower().split(" ")[0]: v
        for k, v in keypoints[cfg.viz.viz_subject].items()
    }
    poses_2d = poses_2d_subj[cfg.viz.viz_action]
    out_poses_2d = poses_2d[cfg.viz.viz_camera]

    poses_3d_subj = {
        k.lower().split(" ")[0]: v
        for k, v in dataset[cfg.viz.viz_subject].items()
    }
    poses_3d = poses_3d_subj[cfg.viz.viz_action]["positions_3d"]
    assert len(poses_3d) == len(poses_2d), "Camera count mismatch"
    out_poses_3d = poses_3d[cfg.viz.viz_camera]

    # NOTE: It seems that data from SemGCN is cropped when compared to the
    # original H36M dataset, which is why we need to make sure here both inputs
    # and outputs have the same sequence length
    L = out_poses_3d.shape[0]
    out_poses_2d = out_poses_2d[:L]

    out_actions = [cfg.viz.viz_camera] * out_poses_2d.shape[0]
    ground_truth = out_poses_3d.copy()
    input_keypoints = out_poses_2d.copy()

    render_loader = DataLoader(
        PoseSequenceGenerator(
            [out_poses_3d],
            [out_poses_2d],
            [out_actions],
            seq_len=cfg.data.seq_len,
            random_start=False,
            drop_last=False,
        ),
        batch_size=cfg.train.batch_size_test,
        shuffle=False,
        num_workers=cfg.train.workers,
        pin_memory=True,
    )

    if cfg.viz.viz_output != "":
        output_name = cfg.viz.viz_output
    else:
        output_name = (
            f"{cfg.viz.viz_subject}_{cfg.viz.viz_action}_"
            f"{cfg.viz.viz_camera}"
        )

    prediction = lift_action(
        render_loader,
        model_pos,
        nsamples=cfg.eval.nsamples,
        strategy=cfg.eval.sample_strategy,
        tot_seq_len=L,
        data_file_path=figures_data_dir / f"{output_name}.npy",
        overide=False,
    )

    # Invert camera transformation
    cam = dataset.cameras()[cfg.viz.viz_subject][cfg.viz.viz_camera]
    prediction = camera_to_world(prediction, R=cam["orientation"], t=0)
    prediction[:, :, 2] -= np.min(prediction[:, :, 2])
    ground_truth = camera_to_world(ground_truth, R=cam["orientation"], t=0)
    ground_truth[:, :, 2] -= np.min(ground_truth[:, :, 2])

    anim_output = {"DiffHPE": prediction, "Ground truth": ground_truth}
    input_keypoints = image_coordinates(
        input_keypoints[..., :2], w=cam["res_w"], h=cam["res_h"]
    )

    output_name = figures_dir / f"{output_name}.gif"

    render_animation(
        input_keypoints,
        anim_output,
        dataset.skeleton(),
        dataset.fps(),
        cfg.viz.viz_bitrate,
        cam["azimuth"],
        str(output_name),
        limit=cfg.viz.viz_limit,
        downsample=cfg.viz.viz_downsample,
        size=cfg.viz.viz_size,
        input_video_path=cfg.viz.viz_video,
        viewport=(cam["res_w"], cam["res_h"]),
        input_video_skip=cfg.viz.viz_skip,
    )


def lift_action(
    data_loader,
    model_pos,
    nsamples,
    strategy,
    tot_seq_len,
    data_file_path,
    overide=False,
):
    if data_file_path.exists() and not overide:
        print("Loading pre-computed predictions.")
        return np.load(data_file_path)

    predictions = evaluate(
        model=model_pos,
        loader=data_loader,
        nsample=nsamples,
        strategy=strategy,
    )[0]
    predictions = torch.cat(predictions, dim=0)
    predictions = predictions.detach().cpu().numpy()
    N, L, J, _ = predictions.shape

    prediction_whole_action = predictions.reshape(N * L, J, 3) / 1000
    return prediction_whole_action[:tot_seq_len, ...]


if __name__ == "__main__":
    main()
