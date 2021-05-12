import os
import sys
sys.path.insert(0, os.getcwd())

import torch

from models.swin_transformer import SwinTransformerBackbone


if __name__ == "__main__":
        
    swin_transformer = SwinTransformerBackbone(
        image_channel=3,
        image_size=224,
        patch_size=4,
        num_channels=96,
        head_dim=32,
        num_layers_in_stages=[2, 2, 6, 2],
        window_size=(7, 7),

    )

    print(swin_transformer)

    x = torch.randn(1, 3, 224, 224)

    print("Input Shape:\n", x.shape)
    print("Output Shape (remain the output size as the input):\n", swin_transformer(x).shape)