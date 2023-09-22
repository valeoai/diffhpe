import torch
import numpy as np


def apply_guidance(x, p_uncond, miss_rate=1.0):
    if p_uncond > 0.0:
        if np.random.rand() < p_uncond:
            if miss_rate == 1.0:
                x = torch.zeros_like(x)
            else:
                # Guidance with random joint masking instead of complete
                # masking
                B, _, J, L = x.shape
                mask = torch.zeros((B, J, L), device=x.device)
                u = np.random.uniform(0.0, 1.0, size=(B, J, L))
                mask[u > miss_rate] = 1.0
                x *= mask[:, None, ...]
    return x
