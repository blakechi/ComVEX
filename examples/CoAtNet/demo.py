import os
import sys

sys.path.insert(0, os.getcwd())

import torch 

from comvex.coatnet import CoAtNetConfig, CoAtNetWithLinearClassifier


if __name__ == "__main__":

    coatnet_config = CoAtNetConfig.CoAtNet_0(num_classes=10, attention_dropout=0.2)
    coatnet = CoAtNetWithLinearClassifier(coatnet_config)

    print(coatnet)

    x = torch.randn(1, 3, 224, 224)

    print("Input Shape:\n", x.shape)
    print("Output Shape:\n", coatnet(x).shape)
