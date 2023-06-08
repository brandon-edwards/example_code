#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import torch


# Diffusion loss
def diffusion_loss(model, samples, noise_scheduler):
    # Sample noise to add to the images
    noise = torch.randn_like(samples)

    # Sample a random timestep for each image
    timesteps = torch.randint(
        0,
        noise_scheduler.num_train_timesteps,
        (samples.shape[0],),
        device=samples.device,
    ).long()

    # Add noise to the clean images according to the noise magnitude at each timestep
    noisy_samples = noise_scheduler.add_noise(samples, noise, timesteps)

    # Predict noise
    noise_pred = model(noisy_samples, timesteps, return_dict=False)[0]

    # MSE loss
    loss = torch.nn.functional.mse_loss(noise_pred, noise)
    return loss
