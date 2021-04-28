import os
import sys
sys.path.insert(0, os.getcwd())

import torch

from models.utils import ResNetWithLinearClassifier, ResNetConfig


if __name__ == "__main__":

    resnet_50_config = ResNetConfig.ResNet_50_ImageNet()
    resnet_50 = ResNetWithLinearClassifier(resnet_50_config)
    print(resnet_50)

    x = torch.randn(1, 3, 1024, 1024)

    print("Input Shape:\n", x.shape)
    print("Output Shape:\n", resnet_50(x).shape)