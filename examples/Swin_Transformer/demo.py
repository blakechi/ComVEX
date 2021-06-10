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
    out = swin_transformer(x)
    print("Input Shape:\n", x.shape)
    print("Output Shape:\n", out.shape)

    out = out.mean()
    out.backward()

    for name, p in swin_transformer.named_parameters():
        print(name, p.grad)