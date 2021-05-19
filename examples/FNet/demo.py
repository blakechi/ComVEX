import os
import sys
sys.path.insert(0, os.getcwd())

import torch

from models.fnet import FNetConfig, FNetWithLinearHead


if __name__ == "__main__":
        
    fnet_config = FNetConfig.FNet_B_12_512(num_classes=1000)
    fnet = FNetWithLinearHead(fnet_config)

    print(fnet)

    x = torch.randn(1, 3, 224, 224)

    print("Input Shape:\n", x.shape)
    print("Output Shape (remain the output size as the input):\n", fnet(x).shape)