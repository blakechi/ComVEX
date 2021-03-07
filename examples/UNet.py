import os
import sys
sys.path.insert(0, os.getcwd())

import torch

from models.utils import UNet
from models.transunet import TransUNet

if __name__ == "__main__":
    unet = UNet(
        input_channel=3,
        middle_channel=1024,
        output_channel=10,
        channel_in_between=[64, 128, 256, 512],
        to_remain_size=True
    )
    print(unet)

    x = torch.randn(1, 3, 572, 572)

    print(unet(x).shape)