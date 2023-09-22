from __future__ import absolute_import
from collections.abc import Iterable

import numpy as np


class Skeleton(object):
    def __init__(
        self,
        parents,
        joints_left,
        joints_right,
        joints_group=None,
        joints_names=None,
    ):
        assert len(joints_left) == len(joints_right)

        self._parents = np.array(parents)
        self._joints_left = joints_left
        self._joints_right = joints_right
        self._joints_group = joints_group
        self._joints_names = joints_names
        if self._joints_names is None:
            self._joints_names = [""] * len(self._parents)
        assert isinstance(self._joints_names, Iterable) and len(
            self._joints_names
        ) == len(
            self._parents
        ), "joint_names should be an iterable with as many elements as joints."
        self._compute_metadata()

    def num_joints(self):
        return len(self._parents)

    def num_bones(self):
        return len(list(filter(lambda x: x >= 0, self._parents)))

    def parents(self):
        return self._parents

    def has_children(self):
        return self._has_children

    def children(self):
        return self._children

    def remove_joints(self, joints_to_remove):
        # Compute list complementary to joints to be removed
        valid_joints = [
            i for i in range(len(self._parents)) if i not in joints_to_remove
        ]

        # Recursive update of parents
        for i in range(len(self._parents)):
            while self._parents[i] in joints_to_remove:
                self._parents[i] = self._parents[self._parents[i]]

        index_offsets = np.zeros(len(self._parents), dtype=int)
        new_parents = []
        for i, parent in enumerate(self._parents):
            if i not in joints_to_remove:
                new_parents.append(parent - index_offsets[parent])
            else:
                index_offsets[i:] += 1
        self._parents = np.array(new_parents)

        # Zip other metadata indexed by joint
        jointwise_metadata = [
            (
                self._joints_names[i],
                i in self._joints_left,  # l/r joints index replaced by mask
                i in self._joints_right,
            )
            for i in range(len(self._joints_names))
        ]

        # Drop zipped entried from higher indices to lower
        joints_to_remove.sort(reverse=True)
        for i_to_pop in joints_to_remove:
            jointwise_metadata.pop(i_to_pop)

        # Unzip
        self._joints_names, ljoints_mask, rjoints_mask = zip(
            *jointwise_metadata
        )

        # Convert l/r masks back to indices
        self._joints_left = [
            i for i, is_left in enumerate(ljoints_mask) if is_left
        ]
        self._joints_right = [
            i for i, is_right in enumerate(rjoints_mask) if is_right
        ]

        # Update other metadata
        self._compute_metadata()

        return valid_joints

    def joints_left(self):
        return self._joints_left

    def joints_right(self):
        return self._joints_right

    def joints_group(self):
        return self._joints_group

    def joints_names(self):
        return self._joints_names

    def bones(self):
        return self._bones

    def bones_left(self):
        return self._bones_left

    def bones_right(self):
        return self._bones_right

    def bones_names(self):
        return self._bones_names

    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)

        # Creates bones as a tuple of (joint, joint_parent) tuples
        self._bones = tuple(
            (j, p) for j, p in enumerate(self._parents) if p >= 0
        )

        self._bones_names = tuple(
            f"{self._joints_names[j]}->{self._joints_names[i]}"
            for i, j in self._bones
        )

        # Creates left and right bones indices list
        bone_parent = dict(self._bones)
        bone_index = {b: i for i, b in enumerate(self._bones)}
        skeleton_left = tuple(
            (j, bone_parent[j]) for j in self._joints_left if j >= 0
        )
        skeleton_right = tuple(
            (j, bone_parent[j]) for j in self._joints_right if j >= 0
        )
        self._bones_left = tuple(bone_index[b] for b in skeleton_left)
        self._bones_right = tuple(bone_index[b] for b in skeleton_right)
