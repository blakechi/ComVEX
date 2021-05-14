import os
import sys
sys.path.insert(0, os.getcwd())

import torch

from models.swin_transformer import SwinTransformerConfig, SwinTransformerWithLinearClassifier


if __name__ == "__main__":
        
    swin_config = SwinTransformerConfig(
        image_channel=3, 
        image_size=224, 
        patch_size=4,
        num_channels=96,
        num_layers_in_stages=[2, 2, 6, 6], 
        head_dim=32,
        window_size=(7, 7),
        shifts=2,
        num_classes=1000,
        use_absolute_position=False,
        use_checkpoint=False,
        use_pre_norm=False, 
        ff_dim=None, 
    )    
    swin_transformer = SwinTransformerWithLinearClassifier(swin_config)

    print(swin_transformer)

    x = torch.randn(1, 3, 224, 224)

    print("Input Shape:\n", x.shape)
    print("Output Shape (remain the output size as the input):\n", swin_transformer(x).shape)