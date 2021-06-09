import os
import sys
sys.path.insert(0, os.getcwd())

import torch

from models.g_mlp import gMLPConfig, gMLPWithLinearClassifier


if __name__ == "__main__":

    gmlp_config = gMLPConfig.gMLP_B(num_classes=1000)
    gmlp = gMLPWithLinearClassifier(gmlp_config)

    print(gmlp)

    x = torch.randn(1, 3, 224, 224)

    print("Input Shape:\n", x.shape)
    print("Output Shape (remain the output size as the input):\n", gmlp(x).shape)