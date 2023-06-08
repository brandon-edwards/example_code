#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import torch, os, argparse, copy, json
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.optim import Adam

from torchvision.transforms import Compose
from torchvision.transforms import ToTensor, Normalize, RandomHorizontalFlip
from torchvision.datasets import CIFAR10

from diffusers.models import UNet2DModel
from diffusers.training_utils import EMAModel
from diffusers import DDPMScheduler, DDPMPipeline

from losses import diffusion_loss
from sampling import evaluate


def create_pytorch_objects(config):
    # Model
    model = get_model(config)

    # Optimizer
    optimizer = get_optim(config, model.parameters())

    # LR scheduler
    scheduler = None

    # Dataloaders
    train_loader, val_loader = get_dataloaders(config)

    # Dummy output for now
    parameters = None

    return model, optimizer, train_loader, val_loader, scheduler, parameters


# Load config file
def get_config(filename):
    with open(filename, "r") as jsonfile:
        config = json.load(jsonfile)

    return config


# Get dataloaders
def get_dataloaders(config):
    # Transform [0, 255] to [-1, 1] and augment
    transforms = Compose(
        [
            ToTensor(),
            Normalize(mean=127.5, std=127.5, inplace=True),
            RandomHorizontalFlip(p=0.5),
        ]
    )

    # Load CIFAR10 datasets
    os.makedirs("./data/train", exist_ok=True)
    os.makedirs("./data/val", exist_ok=True)
    train_dataset = CIFAR10(
        root="./data/train", train=True, transform=transforms, download=True
    )
    val_dataset = CIFAR10(
        root="./data/val", train=False, transform=transforms, download=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)

    return train_loader, val_loader


def get_model(config):
    model = UNet2DModel(
        sample_size=config["data"]["image_size"],  # Sample resolution
        # The number of input channels, 3 for RGB images
        in_channels=config["data"]["channels"],
        out_channels=config["data"]["channels"],
        layers_per_block=2,  # How many ResNet layers to use per UNet block
        # The number of output channels for each UNet block
        block_out_channels=(
            config["model"]["ngf"],
            config["model"]["ngf"],
            2 * config["model"]["ngf"],
            2 * config["model"]["ngf"],
            4 * config["model"]["ngf"],
            4 * config["model"]["ngf"],
        ),
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    return model


# Instantiate and return Adam optimizer
def get_optim(config, params):
    optimizer = Adam(
        params,
        lr=config["optimizer"]["lr"],
        betas=(config["optimizer"]["beta1"], 0.999),
        eps=config["optimizer"]["eps"],
    )
    return optimizer


# Main code
if __name__ == "__main__":
    # Seeding and maintenance
    np.random.seed(2023)
    torch.manual_seed(2023)
    torch.multiprocessing.set_sharing_strategy("file_system")

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    # Get config
    config = get_config("default.json")
    # Populate command line parameters
    config["training"]["batch_size"] = args.batch_size

    # Get objects
    (
        model,
        optimizer,
        train_loader,
        val_loader,
        scheduler,
        parameters,
    ) = create_pytorch_objects(config)

    # Move model to GPU
    model = model.to(config["device"])
    model = torch.nn.DataParallel(model)
    model.train()
    num_parameters = np.sum([np.prod(p.shape) for p in model.parameters()])
    print("Model has %d parameters!" % num_parameters)

    # Get noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["model"]["num_timesteps"]
    )

    # Get EMA helper and validation model
    if config["model"]["ema"]:
        ema_helper = EMAModel(model.parameters(), decay=config["model"]["ema_rate"])
    val_model = copy.deepcopy(model)

    # Logging
    train_dsm_log, val_dsm_log = [], []
    step_idx = 0
    local_dir = config["model"]["log_dir"]

    # For each epoch
    for epoch_idx in range(config["training"]["n_epochs"]):
        model.train()
        # For each batch
        for batch_idx, (images, _) in tqdm(enumerate(iter(train_loader))):
            # Move to device
            images = images.to(config["device"])

            # Get loss
            loss = diffusion_loss(model, images, noise_scheduler)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # EMA
            if config["model"]["ema"]:
                ema_helper.step(model.parameters())

            # Store training loss
            train_dsm_log.append(loss.item())

            # Periodic validation
            if (step_idx + 1) % config["training"]["val_steps"] == 0:
                if config["model"]["ema"]:
                    ema_helper.copy_to(val_model.parameters())
                else:
                    val_model.load_state_dict(model.state_dict())
                val_model.eval()

                with torch.no_grad():
                    # Validate one batch of samples at random noise level
                    images, _ = next(iter(val_loader))
                    # Move to device
                    images = images.to(config["device"])

                    # Get and store loss
                    val_loss = diffusion_loss(val_model, images, noise_scheduler)
                    val_dsm_log.append(val_loss.item())
                # Verbose
                print(
                    "Epoch %d, Step %d, Train. Loss %.3f, Val. Loss %.3f"
                    % (epoch_idx, step_idx, train_dsm_log[-1], val_dsm_log[-1])
                )

            # Increment
            step_idx = step_idx + 1

        # Sample images and save checkpoints periodically
        if (epoch_idx + 1) % config["training"]["save_epochs"] == 0:
            os.makedirs(local_dir, exist_ok=True)
            filename = os.path.join(local_dir, "model_epoch%d.pt" % epoch_idx)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "opt_state": optimizer.state_dict(),
                    "train_dsm_log": train_dsm_log,
                    "val_dsm_log": val_dsm_log,
                    "config": config,
                    "args": args,
                },
                filename,
            )

            # Also run inference and save to file
            if config["model"]["ema"]:
                ema_helper.copy_to(val_model.parameters())
            else:
                val_model.load_state_dict(model.state_dict())
            val_model.eval()

            with torch.no_grad():
                pipeline = DDPMPipeline(unet=val_model, scheduler=noise_scheduler)
                evaluate(config, pipeline, epoch_idx + 1)
