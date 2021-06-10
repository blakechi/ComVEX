import os
import sys
sys.path.insert(0, os.getcwd())

import torch

from comvex.utils import UNet


if __name__ == "__main__":
    
    unet = UNet(
        input_channel=3,
        middle_channel=1024,
        output_channel=2,
        channel_in_between=[64, 128, 256, 512],
    )
    print(unet)

    x = torch.randn(1, 3, 572, 572)

    print("Input Shape:\n", x.shape)
    print("Output Shape (without remaining the output size as the input):\n", unet(x).shape)