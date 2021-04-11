import os
import sys
sys.path.insert(0, os.getcwd())

import torch

from models.vit import ViT


if __name__ == "__main__":

    vit = ViT(
        image_size=224,
        image_channel=1,
        patch_size=16,
        num_classes=2,
        dim=512,
        depth=12,
        num_heads=16,
        ff_dropout=0.0,
        pre_norm=True
    )

    print(vit)

    x = torch.randn(1, 1, 224, 224)

    print("Input Shape:\n", x.shape)
    print("Output Shape:\n", vit(x).shape)