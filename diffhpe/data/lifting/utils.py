from __future__ import absolute_import, division

import numpy as np
from tqdm import tqdm

from .camera import normalize_screen_coordinates, world_to_camera


def create_2d_data(data_path, dataset):
    """Loads keypoints 2D coordinates from disk and normalizes them to camera
    screen.
    """
    keypoints = np.load(data_path, allow_pickle=True)
    keypoints = keypoints["positions_2d"].item()

    for subject in tqdm(keypoints.keys()):
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(
                    kps[..., :2], w=cam["res_w"], h=cam["res_h"]
                )
                keypoints[subject][action][cam_idx] = kps

    return keypoints


def read_3d_data(dataset, subjects_filter=None, action_filter=None):
    """Pass from 3D coordinates in global frame to camera coordinates

    For each subject and action, this function creates a new field
    'positions_3d' inside the dataset object, where it stores a list of arrays
    with 3D coordinates in each camera frame.
    """
    subjects = dataset.subjects()
    if subjects_filter is not None:
        subjects = filter(lambda x: x in subjects_filter, subjects)

    for subject in subjects:
        actions = dataset[subject].keys()
        if action_filter is not None:
            actions = filter(lambda x: x in action_filter, actions)
        for action in tqdm(actions, desc=f"{subject}"):
            anim = dataset[subject][action]

            positions_3d = []
            for cam in anim["cameras"]:
                pos_3d = world_to_camera(
                    anim["positions"],
                    R=cam["orientation"],
                    t=cam["translation"],
                )
                pos_3d[:, :] -= pos_3d[:, :1]  # Remove global offset
                positions_3d.append(pos_3d)
            anim["positions_3d"] = positions_3d

    return dataset


def fetch(
    subjects,
    dataset,
    keypoints,
    action_filter=None,
    stride=1,
    parse_3d_poses=True,
):
    """Extract 2D inputs and 3D outputs from diffhpe.dataset and keypoints objects,
    together with corresponding action (for action recognition for example) and
    camera intrinsics. Filters data according to desired subjects and actions,
    as well as the desired temporal stride.
    """
    out_poses_3d = []
    out_poses_2d = []
    out_actions = []
    out_camera_params = []

    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    # if action.startswith(a):
                    if action.lower().split(" ")[0] == a:
                        found = True
                        break
                if not found:
                    continue

            cams = dataset.cameras()[subject]
            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])
                out_actions.append(
                    [action.split(" ")[0]] * poses_2d[i].shape[0]
                )
                augmented_cam = np.concatenate(
                    [
                        cams[i]["intrinsic"],
                        cams[i]["orientation"],
                        cams[i]["translation"],
                        np.array([i]),
                    ]
                )
                out_camera_params.append(
                    [augmented_cam] * poses_2d[i].shape[0]
                )

            if parse_3d_poses and "positions_3d" in dataset[subject][action]:
                poses_3d = dataset[subject][action]["positions_3d"]
                assert len(poses_3d) == len(poses_2d), "Camera count mismatch"
                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_poses_3d) == 0:
        out_poses_3d = None

    if stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            out_actions[i] = out_actions[i][::stride]
            out_camera_params[i] = out_camera_params[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_poses_3d, out_poses_2d, out_actions, out_camera_params
