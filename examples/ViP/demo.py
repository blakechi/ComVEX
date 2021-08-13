import os
import sys

sys.path.insert(0, os.getcwd())

import torch 

from comvex.vip import ViPConfig, ViPWithLinearClassifier


if __name__ == "__main__":

    vip_config = ViPConfig.ViP_Small_14(num_classes=10, ff_dropout=0.1)
    vip = ViPWithLinearClassifier(vip_config)

    print(vip)

    x = torch.randn(1, 3, 224, 224)

    print("Input Shape:\n", x.shape)
    print("Output Shape:\n", vip(x).shape)
