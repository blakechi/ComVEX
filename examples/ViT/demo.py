import os
import sys
sys.path.insert(0, os.getcwd())

import torch

from comvex.vit import ViTWithLinearClassifier, ViTConfig


if __name__ == "__main__":

    vit_config = ViTConfig.ViT_B(num_classes=10)
    vit = ViTWithLinearClassifier(vit_config)

    print(vit)

    x = torch.randn(1, 3, 224, 224)

    print("Input Shape:\n", x.shape)
    print("Output Shape:\n", vit(x).shape)