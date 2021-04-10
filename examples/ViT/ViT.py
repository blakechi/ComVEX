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
        num_heads=16,
        num_layers=12,
    )