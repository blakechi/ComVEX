import os
import sys
sys.path.insert(0, os.getcwd())

import torch

from comvex.swin_transformer import SwinTransformerConfig, SwinTransformerWithLinearClassifier


if __name__ == "__main__":
        
    swin_config = SwinTransformerConfig.SwinTransformer_B(num_classes=1000)    
    swin_transformer = SwinTransformerWithLinearClassifier(swin_config)

    print(swin_transformer)

    x = torch.randn(1, 3, 224, 224)

    print("Input Shape:\n", x.shape)
    print("Output Shape:\n", swin_transformer(x).shape)