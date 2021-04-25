import os
import sys
sys.path.insert(0, os.getcwd())

import torch

from models.botnet import BoTNetBlock, BoTNetBlockFullPreActivation


if __name__ == "__main__":

    botblock = BoTNetBlockFullPreActivation(32, 64, lateral_size=28, heads=4, stride=2)

    print(botblock)

    x = torch.randn(1, 32, 28, 28)

    print("Input Shape:\n", x.shape)
    print("Output Shape:\n", botblock(x).shape)