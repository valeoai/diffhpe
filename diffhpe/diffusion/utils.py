import math

import numpy as np


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.5):
    # """
    # Create a beta schedule that discretizes the given alpha_t_bar
    # function, which defines the cumulative product of (1-beta) over time
    # from t = [0,1].
    # :param num_diffusion_timesteps: the number of betas to produce.
    # :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
    #                   produces the cumulative product of (1-beta) up to
    #                   that part of the diffusion process.
    # :param max_beta: the maximum beta to use; use values lower than 1 to
    #                  prevent singularities.
    # """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def compute_noise_scheduling(schedule, beta_start, beta_end, num_steps):
    if schedule == "quad":
        beta = (
            np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_steps,
            )
            ** 2
        )
    elif schedule == "linear":
        beta = np.linspace(
            beta_start,
            beta_end,
            num_steps,
        )
    elif schedule == "cosine":
        beta = betas_for_alpha_bar(
            num_steps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            max_beta=beta_end,
        )

    alpha_hat = 1 - beta
    alpha = np.cumprod(alpha_hat)

    sigma = ((1.0 - alpha[:-1]) / (1.0 - alpha[1:]) * beta[1:]) ** 0.5
    return beta, alpha, alpha_hat, sigma
