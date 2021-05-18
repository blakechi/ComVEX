import os
import sys
sys.path.insert(0, os.getcwd())

import torch

from models.fnet import FNetBackbone


if __name__ == "__main__":
        
    fnet = FNetBackbone(
        dim=64,
        depth=3
    )

    print(fnet)

    x = torch.randn(1, 32, 64)

    print("Input Shape:\n", x.shape)
    print("Output Shape (remain the output size as the input):\n", fnet(x).shape)