#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import torch, os
from matplotlib import pyplot as plt


# Sample from VP process
def evaluate(config, pipeline, epoch):
    # Sample
    images = pipeline(
        batch_size=config["sampling"]["batch_size"],
        generator=torch.manual_seed(config["sampling"]["seed"]),
        output_type="numpy",
    ).images

    # Plot and save images
    sample_dir = os.path.join(config["model"]["log_dir"], "samples_epoch%d" % epoch)
    savefile = os.path.join(sample_dir, "samples.png")
    os.makedirs(sample_dir, exist_ok=True)

    plt.figure(figsize=(16, 10))
    for image_idx in range(config["sampling"]["batch_size"]):
        plt.subplot(2, 4, image_idx + 1)
        plt.imshow(images[image_idx])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(savefile, dpi=300)
    plt.close()
