from __future__ import absolute_import, division

import numpy as np
import torch

from diffhpe.utils import wrap
from diffhpe.data.lifting.quaternion import qrot, qinverse


def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the
    # aspect ratio
    return X / w * 2 - [1, h / w]


def image_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Reverse camera frame normalization
    return (X + [1, h / w]) * w / 2


def world_to_camera(X, R, t):
    Rt = wrap(qinverse, False, R)  # Invert rotation
    return wrap(
        qrot, False, np.tile(Rt, X.shape[:-1] + (1,)), X - t
    )  # Rotate and translate


def camera_to_world(X, R, t):
    return wrap(qrot, False, np.tile(R, X.shape[:-1] + (1,)), X) + t


def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original
    MATLAB script.

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    k = camera_params[..., 4:7]
    p = camera_params[..., 7:]

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    r2 = torch.sum(XX[..., :2] ** 2, dim=len(XX.shape) - 1, keepdim=True)

    radial = 1 + torch.sum(
        k * torch.cat((r2, r2**2, r2**3), dim=len(r2.shape) - 1),
        dim=len(r2.shape) - 1,
        keepdim=True,
    )
    tan = torch.sum(p * XX, dim=len(XX.shape) - 1, keepdim=True)

    XXX = XX * (radial + tan) + p * r2

    return f * XXX + c


def project_to_2d_linear(X, camera_params):
    """
    Project 3D points to 2D using only linear parameters (focal length and
    principal point).

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2]
    c = camera_params[..., 2:4]

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)

    return f * XX + c


def uvd2xyz(uvd, f, c, cam_dist):
    """
    transfer uvd to xyz
    :param uvd: N*T*V*3 (uv and z channel)
    :return: root-relative xyz results
    """
    N, T, V, _ = uvd.size()

    dec_out_all = uvd.view(-1, T, V, 3).clone()  # N*T*V*3
    enc_in_all = uvd[:, :, :, :2].view(-1, T, V, 2).clone()  # N*T*V*2

    cam_f_all = f.view(N, T, 1, 1).repeat(1, 1, V, 2)  # N*T*V*2
    cam_c_all = c.view(N, T, 1, 2).repeat(1, 1, V, 1)  # N*T*V*2

    cam_dist = cam_dist.unsqueeze(2).repeat(1, 1, V)  # N*T*V*1

    # change to global
    z_global = dec_out_all[:, :, :, 2] + cam_dist  # N*T*V
    z_global = z_global.unsqueeze(-1)  # N*T*V*1

    uv = enc_in_all - cam_c_all  # N*T*V*2
    xy = -uv * z_global.repeat(1, 1, 1, 2) / cam_f_all  # N*T*V*2
    xyz_global = torch.cat((xy, z_global), -1)  # N*T*V*3
    xyz_offset = xyz_global - xyz_global[:, :, 0, :].unsqueeze(-2).repeat(
        1, 1, V, 1
    )  # N*T*V*3

    return xyz_offset


def uvd2xyz_from_cam(uvd, cam):
    """
    transfer uvd to xyz
    :param uvd: N*T*V*3 (uv and z channel)
    :return: root-relative xyz results
    """
    cam_rot = cam[..., 9:13]
    cam_t = cam[..., 13:16]
    cam_t_in_cam_frame = qrot(qinverse(cam_rot), cam_t)

    return uvd2xyz(
        uvd,
        f=cam[..., 0],
        c=cam[..., 2:4],
        cam_dist=cam_t_in_cam_frame[..., 2],
    )
