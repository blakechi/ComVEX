import os
import sys

sys.path.insert(0, os.getcwd())

import torch 

from comvex.convit import ConViTConfig, ConViTWithLinearClassifier
from comvex.utils import PatchEmbeddingXd
if __name__ == "__main__":

    pe = torch.jit.script(PatchEmbeddingXd(3, 256, 16))
    x = torch.randn(1, 3, 224, 224)
    print(pe(x))
    print(pe.code)

    # convit_config = ConViTConfig.ConViT_B(num_classes=1000, token_dropout=0.2)
    # convit = ConViTWithLinearClassifier(convit_config)

    # print(convit)

    # x = torch.randn(1, 3, 224, 224)

    # print("Input Shape:\n", x.shape)
    # print("Output Shape:\n", convit(x).shape)
