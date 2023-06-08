# EXAMPLE CODE
This is example training code demonstrating what we want to do, current code utilizes GPUs (data parallel is critical to us, our actual data and model is larger)

# Pre-requisites
I tested to create and activate a python3.8 virtual environment; pip install --upgrade pip; pip install -r requirements.txt

# Expectations for Code
With GPUs visible by default, run `python train_ddpm.py --batch_size 32` or whatever batch size is desired.

The first run will create a `data` folder and automatically download CIFAR-10 there.

This current code will crash at the end of the 10th epoch, but training should be observed to work with data parallel across mutliple devices up until that point.


# Configuration
See `default.json` for barebone configurable parameters. Currently this file is used by default.
