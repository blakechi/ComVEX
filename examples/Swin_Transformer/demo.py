import os
import sys
sys.path.insert(0, os.getcwd())

import torch

from models.swin_transformer import SwinTransformerConfig, SwinTransformerWithLinearClassifier


if __name__ == "__main__":
        
    swin_config = SwinTransformerConfig.SwinTransformer_B(
        image_channel=3, 
        image_size=224, 
        num_classes=1000,
        use_absolute_position=False,
        use_checkpoint=False,
    )    
    swin_transformer = SwinTransformerWithLinearClassifier(swin_config)

    print(swin_transformer)

    x = torch.randn(1, 3, 224, 224)

    print("Input Shape:\n", x.shape)
    print("Output Shape (remain the output size as the input):\n", swin_transformer(x).shape)