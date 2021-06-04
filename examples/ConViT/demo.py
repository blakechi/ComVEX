import os
import sys
sys.path.insert(0, os.getcwd())

import torch

from models.convit import ConViTConfig, ConViTWithLinearClassifier
from models.utils.efficient_net.model import SeperateConvXd


if __name__ == "__main__":

    # convit_config = ConViTConfig.ConViT_B(num_classes=1000, token_dropout=0.2)
    # convit = ConViTWithLinearClassifier(convit_config)

    # print(convit)

    # x = torch.randn(1, 3, 224, 224)

    # print("Input Shape:\n", x.shape)
    # print("Output Shape:\n", convit(x).shape)
    s = SeperateConvXd(10, 20, dimension=4)