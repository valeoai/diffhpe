import torch


def measure_bones_length(
    joints_coords: torch.Tensor, skeleton_bones: int
) -> torch.Tensor:
    batch_size, _3, _num_joints, series_length = joints_coords.shape
    num_bones = len(skeleton_bones)
    assert _3 == 3 and _num_joints == num_bones + 1
    bones_lengths = torch.empty(
        (batch_size, num_bones, series_length),
        device=joints_coords.device,
        dtype=joints_coords.dtype,
    )
    for b, (j, p) in enumerate(skeleton_bones):
        bones_lengths[:, b, :] = torch.sum(
            (joints_coords[:, :, j, :] - joints_coords[:, :, p, :]) ** 2,
            axis=1,
        ).sqrt()
    return bones_lengths  # (batch_size, num_bones, series_length)
