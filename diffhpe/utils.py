from __future__ import absolute_import, division

import os
import random
from warnings import warn

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def lr_decay(optimizer, step, lr, decay_step, gamma):
    lr = lr * gamma ** (step / decay_step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def wrap(func, unsqueeze, *args):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """

    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)

    result = func(*args)

    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result


def set_random_seeds(seed, cuda, cudnn_benchmark=None, set_deterministic=None):
    """Set seeds for python random module numpy.random and torch.
    For more details about reproducibility in pytorch see
    https://pytorch.org/docs/stable/notes/randomness.html

    Parameters
    ----------
    seed: int
        Random seed.
    cuda: bool
        Whether to set cuda seed with torch.
    cudnn_benchmark: bool (default=None)
        Whether pytorch will use cudnn benchmark. When set to `None` it will
        not modify torch.backends.cudnn.benchmark (displays warning in the
        case of possible lack of reproducibility). When set to True, results
        may not be reproducible (no warning displayed). When set to False it
        may slow down computations.
    set_deterministic: bool (default=None)
        Whether to refrain from using non-deterministic torch algorithms.
        If you are using CUDA tensors, and your CUDA version is 10.2 or
        greater, you should set the environment variable
        CUBLAS_WORKSPACE_CONFIG according to CUDA documentation:
        https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    Notes
    -----
    In some cases setting environment variable `PYTHONHASHSEED` may be needed
    before running a script to ensure full reproducibility. See
    https://forums.fast.ai/t/solved-reproducibility-where-is-the-randomness-coming-in/31628/14
    Using this function may not ensure full reproducibility of the results as
    we do not set `torch.use_deterministic_algorithms(True)`.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        if isinstance(cudnn_benchmark, bool):
            torch.backends.cudnn.benchmark = cudnn_benchmark
        elif cudnn_benchmark is None:
            if torch.backends.cudnn.benchmark:
                warn(
                    "torch.backends.cudnn.benchmark was set to True which may"
                    " results in lack of reproducibility. In some cases to "
                    "ensure reproducibility you may need to set "
                    "torch.backends.cudnn.benchmark to False.",
                    UserWarning,
                )
        else:
            raise ValueError(
                "cudnn_benchmark expected to be bool or None, "
                f"got '{cudnn_benchmark}'"
            )
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if set_deterministic is None:
        if torch.backends.cudnn.deterministic == False:
            warn(
                "torch.backends.cudnn.deterministic was set to False which may"
                " results in lack of reproducibility. In some cases to "
                "ensure reproducibility you may need to set "
                "torch.backends.cudnn.deterministic to True.",
                UserWarning,
            )
        else:
            torch.use_deterministic_algorithms(set_deterministic)
            if set_deterministic:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Source: https://medium.com/optuna/easy-hyperparameter-management-with-hydra-
# mlflow-and-optuna-783730700e7d


def log_params_from_omegaconf_dict(params, mlflow_on):
    if mlflow_on:
        for param_name, element in params.items():
            _explore_recursive(param_name, element)


def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f"{parent_name}.{k}", v)
            else:
                # Supposes lazy mlflow import in main script
                mlf.log_param(f"{parent_name}.{k}", v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            # Supposes lazy mlflow import in main script
            mlf.log_param(f"{parent_name}.{i}", v)
