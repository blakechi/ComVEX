import os
import sys

sys.path.insert(0, os.getcwd())

import torch 

from comvex.convit import ConViTConfig, ConViTWithLinearClassifier
from comvex.utils.rand_augment import RandAugment

if __name__ == "__main__":

    randa = torch.jit.script(RandAugment(3, 5))
    x = torch.randn(1, 3, 224, 224)
    y = randa(x)
    print(y)

    # convit_config = ConViTConfig.ConViT_B(num_classes=1000, token_dropout=0.2)
    # convit = ConViTWithLinearClassifier(convit_config)

    # print(convit)

    # x = torch.randn(1, 3, 224, 224)

    # print("Input Shape:\n", x.shape)
    # print("Output Shape:\n", convit(x).shape)
