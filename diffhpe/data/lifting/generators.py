from __future__ import absolute_import, print_function

import math
from functools import reduce

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class PoseGenerator(Dataset):
    """Pytorch Dataset wrapper around fetched pose data"""

    def __init__(self, poses_3d, poses_2d, actions, cam):
        assert poses_3d is not None

        self._poses_3d = np.concatenate(poses_3d)
        self._poses_2d = np.concatenate(poses_2d)
        self._cam = np.concatenate(cam)

        self._actions = reduce(lambda x, y: x + y, actions)

        assert (
            self._poses_3d.shape[0] == self._poses_2d.shape[0]
            and self._poses_3d.shape[0] == len(self._actions)
            and self._poses_3d.shape[0] == self._cam.shape[0]
        )
        print("Generating {} poses...".format(len(self._actions)))

    def __getitem__(self, index):
        out_pose_3d = self._poses_3d[index]
        out_pose_2d = self._poses_2d[index]

        out_pose_3d = torch.from_numpy(out_pose_3d).float()
        out_pose_2d = torch.from_numpy(out_pose_2d).float()

        return out_pose_3d, out_pose_2d

    def __len__(self):
        return len(self._actions)


class PoseSequenceGenerator(Dataset):
    """Pytorch Dataset wrapper around fetched pose data capable of sampling
    pose sequences
    """

    possible_miss_types_rates = {
        "no_miss": 0.2,
        "random": 0.2,
        "random_left_arm_right_leg": 0.4,
        "structured_joint": 0.4,
        "structured_frame": 0.2,
        # "noisy",  # not including 'noisy' in the sampling for now
    }

    def __init__(
        self,
        poses_3d,
        poses_2d,
        cameras,
        seq_len=8,
        random_start=False,
        drop_last=True,
        miss_type="no_miss",
        miss_rate=0.2,
        noise_sigma=5,  # TODO: Check std of CPN error
    ):
        assert poses_3d is not None

        self._seq_len = seq_len
        self._random_start = random_start
        self._poses_3d = poses_3d
        self._poses_2d = poses_2d
        self._cameras = cameras
        self.drop_last = drop_last
        self.miss_type = miss_type
        self.miss_rate = miss_rate
        self.noise_sigma = noise_sigma

        assert len(self._poses_3d) == len(self._poses_2d)
        print("Generating {} poses...".format(len(self._poses_3d)))

        # Prepare tables mapping iterator index to action videos and frame
        # indices corresponding to sequence start within each video
        self._map_index_to_pose = []
        self._map_index_to_frame = []
        self._ds_len = 0

        for i, pose in enumerate(self._poses_3d):
            pose_size = pose.shape[0] // self._seq_len
            # Handle end of sequence smaller than window size
            if not drop_last:
                last_seq_size = pose.shape[0] % self._seq_len
                if last_seq_size > 0:
                    pose_size += 1
            pose_indices = [i] * pose_size
            self._map_index_to_pose += pose_indices
            frame_indices = [k * self._seq_len for k in range(pose_size)]
            self._map_index_to_frame += frame_indices
            self._ds_len += pose_size

    def __getitem__(self, index):
        # Get action video corresponding to index
        pose_index = self._map_index_to_pose[index]

        out_pose_3d = self._poses_3d[pose_index]
        out_pose_2d = self._poses_2d[pose_index]
        out_camera = self._cameras[pose_index]

        out_pose_3d = torch.from_numpy(out_pose_3d)
        out_pose_2d = torch.from_numpy(out_pose_2d)
        out_camera = torch.from_numpy(np.array(out_camera))

        # Get index of sequence start within video, ...
        if self._random_start:
            # ... either by randomly sampling it at random (which is useful
            # for at training to increase sequence diversity)
            pose_len = out_pose_3d.shape[0]

            seq_start = torch.randint(
                low=0, high=pose_len - self._seq_len, size=(1,)
            ).item()
        else:
            # ... or by fetching the non-overlappin sequence start in correct
            # order
            seq_start = self._map_index_to_frame[index]

        seq_end = seq_start + self._seq_len

        # Pad sequences when last window is smaller than window size
        if not self.drop_last and seq_end > out_pose_3d.shape[0]:
            n_pads = seq_end - out_pose_3d.shape[0]
            tot_length, J, _ = out_pose_3d.shape
            out_pose_3d = F.pad(
                out_pose_3d[None, ...],
                (0, 0, 0, 0, 0, n_pads),
                mode="replicate",
            ).reshape(tot_length + n_pads, J, 3)
            out_pose_2d = F.pad(
                out_pose_2d[None, ...],
                (0, 0, 0, 0, 0, n_pads),
                mode="replicate",
            ).reshape(tot_length + n_pads, J, 2)
            out_camera = torch.from_numpy(
                np.array([out_camera[0].item()] * (tot_length + n_pads))
            )
        pose_3d = out_pose_3d[seq_start:seq_end]  # L,J,3
        pose_2d = out_pose_2d[seq_start:seq_end]  # L,J,2
        cam = out_camera[seq_start:seq_end]  # L

        pose_seq_shape = (self._seq_len, pose_2d.shape[1])

        # Whether to sample an the occlusion pattern uniformly or set it
        if self.miss_type == "all":
            miss_type = np.random.choice(
                list(self.possible_miss_types_rates.keys())
            )
            miss_rate = self.possible_miss_types_rates[miss_type]
        else:
            miss_type = self.miss_type
            miss_rate = self.miss_rate

        if miss_type == "no_miss":
            mask = np.ones(pose_seq_shape)
        elif miss_type == "random":
            # Random Missing Joints in Random Frames
            mask = np.zeros(pose_seq_shape)
            u = np.random.uniform(0.0, 1.0, size=pose_seq_shape)
            mask[u > miss_rate] = 1.0
        elif miss_type == "random_left_arm_right_leg":
            # Left Arm and Right Leg Random Missing
            mask = np.ones(pose_seq_shape)
            rand = np.random.choice(
                self._seq_len,
                size=math.floor(miss_rate * self._seq_len),
                replace=False,
            ).tolist()
            # TODO: make compatible with other skels?
            left_arm_right_leg = [1, 2, 3, 11, 12, 13]
            for i in left_arm_right_leg:
                mask[rand, i] = 0.0
        elif miss_type == "structured_joint":
            # Structured Joint Missing
            mask = np.ones(pose_seq_shape)
            occl_len = int(self._seq_len * miss_rate)
            rand = np.random.choice(
                self._seq_len - occl_len, size=1, replace=False
            )
            right_leg = [1, 2, 3]  # TODO: make compatible with other skels?

            mask[rand[0] : rand[0] + occl_len, right_leg] = 0.0
        elif miss_type == "structured_frame":
            # Structured Frame Missing
            mask = np.ones(pose_seq_shape)
            occl_len = int(self._seq_len * miss_rate)
            rand = np.random.choice(
                self._seq_len - occl_len, size=1, replace=False
            )
            mask[rand[0] : rand[0] + occl_len] = 0.0
        elif miss_type == "noisy":
            # Noisy
            mask = np.ones(pose_seq_shape)
            noise = np.random.normal(0, self.noise_sigma, size=pose_2d.shape)
            pose_2d += noise
        else:
            raise ValueError(f"Unexpected miss_type: {self.miss_type}")

        return {
            "pose_3d": pose_3d,
            "pose_2d": pose_2d * mask[..., None],
            "cam": cam,
        }

    def __len__(self):
        return self._ds_len
