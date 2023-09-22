import os
import pickle
from pathlib import Path
from contextlib import nullcontext

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffhpe.data.lifting.generators import PoseSequenceGenerator
from diffhpe.data.lifting.h36m_lifting import (
    TEST_SUBJECTS,
    TRAIN_SUBJECTS,
    Human36mDataset,
)
from diffhpe.data.lifting.camera import project_to_2d
from diffhpe.data.lifting.utils import create_2d_data, fetch, read_3d_data
from diffhpe.diffusion.diff_lifting import LiftingDiffusionModel
from diffhpe.metrics import (
    coordwise_error,
    jointwise_error,
    mpjpe_error,
    sagittal_symmetry,
    sagittal_symmetry_per_bone,
    segments_time_consistency,
    segments_time_consistency_per_bone,
)
from diffhpe.supervised.lifting import LiftingSupervisedDebugModel
from diffhpe.utils import (
    log_params_from_omegaconf_dict,
    seed_worker,
    set_random_seeds,
)


def save_csv_log(
    output_dir,
    head,
    value,
    is_create=False,
    file_name="test",
    log_in_mlf=False,
):
    if len(value.shape) < 2:
        value = np.expand_dims(value, axis=0)
    df = pd.DataFrame(value)
    file_path = f"{output_dir}/{file_name}.csv"
    if not os.path.exists(file_path) or is_create:
        df.to_csv(file_path, header=head, index=False)
    else:
        with open(file_path, "a") as f:
            df.to_csv(f, header=False, index=False)
    if log_in_mlf:
        # Lazily imported
        mlf.log_artifact(file_path)


def save_state(
    model,
    optimizer,
    scheduler,
    epoch_no,
    foldername,
    log_in_mlf=False,
    tag=None,
):
    if tag is not None:
        tag = f"_{tag}"
    else:
        tag = ""

    params = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch_no,
    }
    torch.save(model.state_dict(), f"{foldername}/model{tag}.pth")
    torch.save(params, f"{foldername}/params{tag}.pth")
    if log_in_mlf:
        # Lazily imported
        mlf.log_artifact(f"{foldername}/model{tag}.pth")


def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    foldername="",
    log_in_mlf=False,
):
    valid_epoch_interval = config.valid_epoch_interval
    mpjpe_epoch_interval = config.mpjpe_epoch_interval
    load_state = config.resume != ""

    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=1e-6)
    if load_state:
        optimizer.load_state_dict(
            torch.load(f"{config.resume}/params.pth")["optimizer"]
        )

    lr_scheduler_type = config.lr_scheduler
    if lr_scheduler_type == "cosine":
        T_max = config.epochs // config.n_annealing
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=config.lr_min,
        )
    elif lr_scheduler_type == "plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=0.5,
            min_lr=config.lr_min,
            patience=config.lr_patience,
            threshold=config.lr_threshold,
        )
    else:
        raise ValueError(
            "Accepted lr_scheduler values are 'cosine' and 'plateau'."
            f"Got {lr_scheduler_type}."
        )
    if load_state and not config.restart_scheduler:
        lr_scheduler.load_state_dict(
            torch.load(f"{config.resume}/params.pth")["scheduler"]
        )

    train_loss = []
    valid_loss = []
    train_loss_epoch = []
    valid_loss_epoch = []

    best_valid_loss = 1e10
    best_mpjpe = 1e10
    start_epoch = 0
    if load_state:
        start_epoch = torch.load(f"{config.resume}/params.pth")["epoch"]

    for epoch_no in range(start_epoch, config.epochs):
        avg_loss = 0
        model.train()
        optimizer.zero_grad()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                loss = model(train_batch).mean()
                loss.backward()
                avg_loss += loss.item()
                if not config.grad_cumul or (
                    batch_no % config.grad_cumul_n == 0
                    or batch_no == len(train_loader)
                ):
                    optimizer.step()
                    optimizer.zero_grad()

                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
        epoch_loss_tr = avg_loss / batch_no
        train_loss.append(epoch_loss_tr)
        train_loss_epoch.append(epoch_no)

        metrics_to_log = {
            "tr_loss": epoch_loss_tr,
        }

        if (
            valid_loader is not None
            and (epoch_no + 1) % valid_epoch_interval == 0
        ):
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(
                    valid_loader, mininterval=5.0, maxinterval=50.0
                ) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0).mean()
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid
                                / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )

            epoch_loss_val = avg_loss_valid / batch_no
            valid_loss.append(epoch_loss_val)
            valid_loss_epoch.append(epoch_no)
            metrics_to_log["val_loss"] = epoch_loss_val

            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    epoch_loss_val,
                    "at",
                    epoch_no,
                )
                save_state(
                    model,
                    optimizer,
                    lr_scheduler,
                    epoch_no,
                    foldername,
                    log_in_mlf=log_in_mlf,
                    tag="best_val",
                )
                # Log when validation loss improves
                metrics_to_log.update(
                    {
                        "best_epoch_loss": epoch_no,
                        "best_val_loss": epoch_loss_val,
                    }
                )

            if lr_scheduler_type == "plateau":
                lr_scheduler.step(best_valid_loss)
            else:
                lr_scheduler.step()

            if (epoch_no + 1) == config.epochs:
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.plot(train_loss_epoch, train_loss)
                ax.plot(valid_loss_epoch, valid_loss)
                ax.grid(True)
                plt.show()
                fig.savefig(f"{foldername}/loss.png")

        # Compute MPJPE every desired nb of epochs (costly operation!!)
        if (
            valid_loader is not None
            and (epoch_no + 1) % mpjpe_epoch_interval == 0
        ):
            __, _, res = evaluate(
                model=model,
                loader=valid_loader,
                nsample=5,
                strategy="average",
            )
            mpjpe_val = res["mpjpe"]
            metrics_to_log["val_mpjpe"] = mpjpe_val

            # Log to MLFlow when there is an improvement!
            if best_mpjpe > mpjpe_val:
                best_mpjpe = mpjpe_val
                metrics_to_log.update(
                    {
                        "best_epoch_mpjpe": epoch_no,
                        "best_val_mpjpe": best_mpjpe,
                    }
                )
                save_state(
                    model,
                    optimizer,
                    lr_scheduler,
                    epoch_no,
                    foldername,
                    log_in_mlf=log_in_mlf,
                    tag="best_mpjpe",
                )

        # Write all logs into MLflow at once
        if log_in_mlf:
            # Lazily imported
            mlf.log_metrics(
                metrics_to_log,
                step=epoch_no,
            )

    save_state(
        model, optimizer, lr_scheduler, config.epochs, foldername, tag="end"
    )
    np.save(f"{foldername}/train_loss.npy", np.array(train_loss))
    np.save(f"{foldername}/valid_loss.npy", np.array(valid_loss))
    return best_mpjpe


def evaluate(model, loader, nsample=5, strategy="best"):
    assert strategy in ["best", "average", "reproj", "worst", "all"], (
        f"Invalid strategy: {strategy}. Possible values are 'best', 'average'"
        "'worst', 'reproj', or 'all'"
    )

    with torch.no_grad():
        model.eval()
        mpjpe_total = 0

        all_target = []
        all_generated_samples = []

        if strategy == "all":
            m_p3d_h36 = {"average": 0, "best": 0, "worst": 0, "reproj": 0}
        else:
            m_p3d_h36 = {strategy: 0}

        n = 0

        with tqdm(loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                # (B,nsample,3,J,L)
                if isinstance(model, nn.DataParallel):
                    samples = model.module.evaluate(test_batch, nsample)
                else:
                    samples = model.evaluate(test_batch, nsample)
                batch_size, _, _, J, L = samples.shape

                n += batch_size

                samples = samples.permute(0, 1, 4, 3, 2)  # (B,nsample,L,J,3)

                # (B,L,J,3)
                target_3d = test_batch["pose_3d"].to(samples.device)
                input_2d = test_batch["pose_2d"].to(samples.device)
                input_cam = test_batch["cam"].to(samples.device)

                if strategy != "all":
                    (
                        renorm_pred_pose,
                        mpjpe_current,
                        mpjpe_p3d_h36,
                    ) = evaluation_metrics(
                        samples=samples,
                        target_3d=target_3d,
                        input_2d=input_2d,
                        input_cam=input_cam,
                        strategy=strategy,
                    )

                    all_generated_samples.append(renorm_pred_pose)

                    mpjpe_total += mpjpe_current.item()
                    m_p3d_h36[strategy] += mpjpe_p3d_h36.cpu().data.numpy()
                else:
                    for strat in ["average", "best", "worst", "reproj"]:
                        (
                            renorm_pred_pose,
                            mpjpe_current,
                            mpjpe_p3d_h36,
                        ) = evaluation_metrics(
                            samples=samples,
                            target_3d=target_3d,
                            input_2d=input_2d,
                            input_cam=input_cam,
                            strategy=strat,
                        )

                        # Only update progressbar with MPJPE of average pose
                        if strat == "average":
                            mpjpe_total += mpjpe_current.item()
                            all_generated_samples.append(renorm_pred_pose)
                        m_p3d_h36[strat] += mpjpe_p3d_h36.cpu().data.numpy()

                all_target.append(target_3d)

                it.set_postfix(
                    ordered_dict={
                        "average_mpjpe": mpjpe_total / batch_no,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            print("Average MPJPE:", mpjpe_total / batch_no)

            if strategy == "all":
                ret = {
                    "mpjpe_avg": m_p3d_h36["average"] / (n * L * J),
                    "mpjpe_best": m_p3d_h36["best"] / (n * L * J),
                    "mpjpe_worst": m_p3d_h36["worst"] / (n * L * J),
                    "mpjpe_reproj": m_p3d_h36["reproj"] / (n * L * J),
                }
            else:
                ret = {"mpjpe": m_p3d_h36[strategy] / (n * L * J)}

            return all_generated_samples, all_target, ret


def reproj_pose(samples, gt_3d, gt_2d, cam_params):
    batch_size, nsample, seq_len, njoints, _ = samples.shape
    gt_3d_traj = gt_3d[:, :, :1, :]  # 3d trajectory of root joint
    gt_3d_traj = gt_3d_traj.unsqueeze(1).repeat(1, nsample, 1, 1, 1)
    samples += gt_3d_traj  # move predictions along root trajectory

    cam_params = cam_params[..., :9]

    reproj_samples = project_to_2d(  # (B, S, L, J, 2)
        samples.reshape(batch_size * nsample * seq_len, njoints, 3),
        camera_params=cam_params.reshape(
            batch_size * seq_len,
            -1,
        ).repeat(nsample, 1),
    ).reshape(batch_size, nsample, seq_len, njoints, 2)

    # (B,S,L,J,1)
    errors_2d = torch.norm(
        reproj_samples - gt_2d[:, None, ...], dim=4, keepdim=True
    )
    selected_ind = torch.argmin(errors_2d, dim=1, keepdim=True)  # (B,1,L,J,1)
    return torch.gather(
        samples, dim=1, index=selected_ind.repeat(1, 1, 1, 1, 3)
    ).reshape(batch_size, seq_len, njoints, 3)


def best_or_worst_pose(samples, gt_3d, strategy):
    batch_size, nsample, L, J, _ = samples.shape
    jw_error = jointwise_error(  # (B*N*L, J)
        samples,
        gt_3d.unsqueeze(1).repeat(1, nsample, 1, 1, 1),
        mode="no_agg",
    )
    error = torch.mean(jw_error.reshape(-1, L * J), dim=1)  # (B*N,)
    error = error.reshape(batch_size, nsample)  # (B, N)

    # Best/worst samples
    statistic = torch.argmin if strategy == "best" else torch.argmax
    poses_idx = statistic(error, dim=1, keepdim=True)  # (B,)

    return torch.gather(
        samples,
        dim=1,
        index=poses_idx[:, :, None, None, None].repeat(1, 1, L, J, 3),
    ).reshape(batch_size, L, J, 3)


def evaluation_metrics(
    samples,
    target_3d,
    input_2d,
    input_cam,
    strategy,
):
    batch_size, nsample, L, J, _ = samples.shape

    if strategy == "best" or strategy == "worst":
        pred_pose = best_or_worst_pose(
            samples=samples,
            gt_3d=target_3d,
            strategy=strategy,
        )
    elif strategy == "reproj":
        pred_pose = reproj_pose(
            samples=samples,
            gt_3d=target_3d,
            gt_2d=input_2d,
            cam_params=input_cam,
        )
    else:
        # This picks the average pose over nsample samples from the
        # model
        pred_pose = torch.mean(samples, axis=1)

    mpjpe_p3d_h36 = mpjpe_error(
        pred_pose.view(batch_size, L, J, 3),
        target_3d.view(batch_size, L, J, 3),
        mode="sum",
    )

    mpjpe_current = mpjpe_error(
        pred_pose.view(batch_size, L, J, 3),
        target_3d.view(batch_size, L, J, 3),
        mode="average",
    )

    return pred_pose * 1000, mpjpe_current * 1000, mpjpe_p3d_h36 * 1000


def fetch_and_prepare_data(cfg):
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

    print("==> Loading 2D detections...")
    inputs_path = data_dir / (
        f"data_2d_{cfg.data.dataset}_{cfg.data.keypoints}.npz"
    )
    keypoints = create_2d_data(inputs_path, dataset)
    return keypoints, dataset


def get_subjects_and_actions(dataset, cfg):
    if cfg.data.use_valid:
        subjects_train = TRAIN_SUBJECTS[:-1]
        subjects_val = TRAIN_SUBJECTS[-1:]
    else:
        subjects_train = TRAIN_SUBJECTS
        subjects_val = []
    subjects_test = TEST_SUBJECTS

    if cfg.data.data == "one":
        subjects_train = subjects_train[0]

    action_filter = (
        None if cfg.data.actions == "*" else cfg.data.actions.split(",")
    )
    if action_filter is not None:
        action_filter = map(
            lambda x: dataset.define_actions(x)[0], action_filter
        )
        print("==> Selected actions: {}".format(action_filter))

    return [subjects_train, subjects_val, subjects_test], action_filter


def create_dataloader(
    keypoints,
    dataset,
    action_filter,
    subjects,
    cfg,
    train=True,
):
    poses, poses_2d, _, cameras = fetch(
        subjects,
        dataset,
        keypoints,
        action_filter,
    )
    generator = PoseSequenceGenerator(
        poses,
        poses_2d,
        cameras,
        seq_len=cfg.data.seq_len,
        random_start=train,
        miss_type=cfg.data.miss_type,
        miss_rate=cfg.data.miss_rate,
    )

    data_loader = DataLoader(
        generator,
        batch_size=(
            cfg.train.batch_size if train else cfg.train.batch_size_test
        ),
        shuffle=True if train else False,
        num_workers=cfg.train.workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
    )
    return data_loader


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("==> Using settings:")
    print(OmegaConf.to_yaml(cfg))

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

    cwd = Path(os.getcwd())
    output_dir = cwd / "results" / cfg.run.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpts_dir = Path(os.getcwd()) / cfg.run.output_dir
    ckpts_dir.mkdir(exist_ok=True)
    seq_len = cfg.data.seq_len

    # Load preprocessed or raw dataset
    keypoints, dataset = fetch_and_prepare_data(cfg)

    # Split subjects and prepare actions
    subjects_split, action_filter = get_subjects_and_actions(dataset, cfg)
    subjects_train, subjects_val, subjects_test = subjects_split

    # Set seeds for init reproducibility
    print(f"==> Setting seeds to {cfg.run.seed} for init")
    set_random_seeds(
        seed=cfg.run.seed,
        cuda=True,
        cudnn_benchmark=cfg.run.cudnn_benchmark,
        set_deterministic=cfg.run.set_deterministic,
    )

    mlflow_on = cfg.run.mlflow_on
    if mlflow_on:
        # Lazy import of MLFlow if requested
        import mlflow as mlf
        mlf.set_tracking_uri(f"file://{cfg.run.mlflow_uri}/mlruns")
        mlf.set_experiment(cfg.run.output_dir)

    # Used to log to MLFlow or not depending on config
    context = mlf.start_run if mlflow_on else nullcontext

    if cfg.run.mode == "train":
        train_loader = create_dataloader(
            keypoints=keypoints,
            dataset=dataset,
            action_filter=action_filter,
            subjects=subjects_train,
            cfg=cfg,
            train=True,
        )
        print(
            ">>> Training dataset length: {:d}".format(
                len(train_loader) * cfg.train.batch_size
            )
        )

        valid_loader = create_dataloader(
            keypoints=keypoints,
            dataset=dataset,
            action_filter=action_filter,
            subjects=subjects_val if cfg.data.use_valid else subjects_test,
            cfg=cfg,
            train=False,
        )
        print(
            ">>> Validation dataset length: {:d}".format(
                len(valid_loader) * cfg.train.batch_size_test
            )
        )

        # Creating model
        model = ModelClass(
            cfg,
            device,
            skeleton=dataset.skeleton(),
            seq_len=seq_len,
        )

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        model.to(device)

        with context():
            log_params_from_omegaconf_dict(cfg, mlflow_on)
            best_valid_mpjpe = train(
                model,
                cfg.train,
                train_loader,
                valid_loader=valid_loader,
                foldername=ckpts_dir,
                log_in_mlf=mlflow_on,
            )
            return best_valid_mpjpe
    elif cfg.run.mode == "test":
        with context():
            log_params_from_omegaconf_dict(cfg, mlflow_on)
            actions = [
                "walking",
                "eating",
                "smoking",
                "discussion",
                "directions",
                "greeting",
                "phoning",
                "posing",
                "purchases",
                "sitting",
                "sittingdown",
                "photo",
                "waiting",
                "walkdog",
                "walktogether",
            ]

            model = ModelClass(
                cfg,
                device,
                skeleton=dataset.skeleton(),
                seq_len=seq_len,
            )

            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                model = nn.DataParallel(model)

            model.to(device)
            # TODO: Should we keep/use the model_l arg here?
            model_l_path = cwd / cfg.eval.model_l
            model.load_state_dict(torch.load(f"{model_l_path}/model.pth"))

            if cfg.eval.sample_strategy == "all":
                head = np.array(
                    [
                        "act",
                        "mpjpe_avg",
                        "sag sym",
                        "seg std",
                        "mpjpe_best",
                        "mpjpe_worst",
                        "mpjpe_reproj",
                    ]
                )
                errs = np.zeros([len(actions) + 1, 6])
            else:
                head = np.array(["act", "mpjpe", "sag sym", "seg std"])
                errs = np.zeros([len(actions) + 1, 3])

            analytics = {
                k: (
                    np.zeros(
                        [len(actions) + 1, dataset.skeleton().num_bones()]
                    ),
                    ["act", *dataset.skeleton().bones_names()],
                )
                for k in ["seg_symmetry", "seg_consistency"]
            }
            analytics["cw_err"] = (
                np.zeros([len(actions) + 1, 3]),
                ["act", "x", "y", "z"],
            )
            analytics["jw_err"] = (
                np.zeros([len(actions) + 1, dataset.skeleton().num_joints()]),
                ["act", *dataset.skeleton().joints_names()],
            )

            for i, action in enumerate(actions):
                print(f"Assessing action: {action} - [{i + 1}/{len(actions)}]")
                test_loader = create_dataloader(
                    keypoints=keypoints,
                    dataset=dataset,
                    action_filter=[action],
                    subjects=subjects_test,
                    cfg=cfg,
                    train=False,
                )
                print(
                    ">>> Test dataset length: {:d}".format(
                        test_loader.__len__() * cfg.train.batch_size_test
                    )
                )

                generated_poses, target_poses, ret = evaluate(
                    model,
                    test_loader,
                    nsample=cfg.eval.nsamples,
                    strategy=cfg.eval.sample_strategy,
                )

                errs[i, 0] = ret.get("mpjpe", ret.get("mpjpe_avg"))

                with torch.no_grad():
                    generated_poses = torch.cat(
                        generated_poses, dim=0
                    ).permute(0, 3, 2, 1)
                    errs[i, 1] = (
                        sagittal_symmetry(
                            joints_coords=generated_poses,
                            skeleton=dataset.skeleton(),
                            mode="average",
                            squared=False,
                        )
                        .cpu()
                        .numpy()
                    )
                    errs[i, 2] = (
                        segments_time_consistency(
                            joints_coords=generated_poses,
                            skeleton=dataset.skeleton(),
                            mode="std",
                        )
                        .cpu()
                        .numpy()
                    )

                    if "mpjpe_best" in ret:
                        errs[i, 3] = ret["mpjpe_best"]
                    if "mpjpe_worst" in ret:
                        errs[i, 4] = ret["mpjpe_worst"]
                    if "mpjpe_reproj" in ret:
                        errs[i, 5] = ret["mpjpe_reproj"]

                    target_poses = torch.cat(target_poses, dim=0)

                    bw_seg_sym = (
                        sagittal_symmetry_per_bone(
                            joints_coords=generated_poses,
                            skeleton=dataset.skeleton(),
                            mode="average",
                            squared=False,
                        )
                        .cpu()
                        .numpy()
                    )

                    analytics["seg_symmetry"][0][
                        i, dataset.skeleton().bones_left()
                    ] = bw_seg_sym
                    analytics["seg_symmetry"][0][
                        i, dataset.skeleton().bones_right()
                    ] = bw_seg_sym

                    analytics["seg_consistency"][0][i] = (
                        segments_time_consistency_per_bone(
                            joints_coords=generated_poses,
                            skeleton=dataset.skeleton(),
                            mode="std",
                        )
                        .cpu()
                        .numpy()
                    )
                    analytics["jw_err"][0][i] = (
                        jointwise_error(
                            generated_poses.permute(0, 3, 2, 1),
                            target_poses,
                            "average",
                        )
                        .cpu()
                        .numpy()
                    )
                    analytics["cw_err"][0][i] = (
                        coordwise_error(
                            generated_poses.permute(0, 3, 2, 1),
                            target_poses,
                            "average",
                        )
                        .cpu()
                        .numpy()
                    )

            errs[-1] = np.mean(errs[:-1], axis=0)
            if mlflow_on:
                if "mpjpe_avg" in ret:
                    mlf.log_metric(
                        "best_val_avg_mpjpe",
                        errs[-1, 0],
                    )
                else:
                    mlf.log_metric(
                        "best_val_mpjpe",
                        errs[-1, 0],
                    )
                mlf.log_metric(
                    "sag_sym",
                    errs[-1, 1],
                )
                mlf.log_metric(
                    "seg_std",
                    errs[-1, 2],
                )
                if "mpjpe_best" in ret:
                    mlf.log_metric(
                        "best_val_best_mpjpe",
                        errs[-1, 3],
                    )
                if "mpjpe_worst" in ret:
                    mlf.log_metric(
                        "best_val_worst_mpjpe",
                        errs[-1, 4],
                    )
                if "mpjpe_reproj" in ret:
                    mlf.log_metric(
                        "best_val_reproj_mpjpe",
                        errs[-1, 5],
                    )
            actions = np.array(actions + ["average"])[:, None]
            value = np.hstack([actions, errs.astype(np.str)])
            save_csv_log(
                output_dir=output_dir,
                head=head,
                value=value,
                is_create=True,
                file_name="protocol_1_err",
                log_in_mlf=mlflow_on,
            )

            for metric_name, (values, a_head) in analytics.items():
                values[-1] = np.mean(values[:-1], axis=0)
                values = np.hstack([actions, values.astype(np.str)])
                save_csv_log(
                    output_dir=output_dir,
                    head=a_head,
                    value=values,
                    is_create=True,
                    file_name=metric_name,
                    log_in_mlf=mlflow_on,
                )


if __name__ == "__main__":
    main()
