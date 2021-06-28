import os
import sys

sys.path.insert(0, os.getcwd())

import torch 

from comvex.cait import CaiTConfig, CaiTWithLinearClassifier
from comvex.xcit.model import PatchProjection

if __name__ == "__main__":
    p = PatchProjection(3, 16, 256)
    print(p)

    # cait_config = CaiTConfig.CaiT_XXS_24(num_classes=1000)
    # cait = CaiTWithLinearClassifier(cait_config)

    # print(cait)

    # x = torch.randn(1, 3, 224, 224)

    # print("Input Shape:\n", x.shape)
    # print("Output Shape:\n", cait(x).shape)
