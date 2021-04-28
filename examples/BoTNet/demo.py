import os
import sys
sys.path.insert(0, os.getcwd())

import torch

from models.botnet import BoTNetWithLinearClassifier, BoTNetConfig


if __name__ == "__main__":
    botnet_50_config = BoTNetConfig.BoTNet_50_ImageNet()

    botnet_50 = BoTNetWithLinearClassifier(botnet_50_config)
    print(botnet_50)

    x = torch.randn(1, 3, 1024, 1024)

    print("Input Shape:\n", x.shape)
    print("Output Shape:\n", botnet_50(x).shape)