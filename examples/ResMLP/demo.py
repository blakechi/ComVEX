import os
import sys
sys.path.insert(0, os.getcwd())

import torch

from models.resmlp import ResMLPConfig, ResMLPWithLinearClassifier


if __name__ == "__main__":
    
    resmlp_config = ResMLPConfig.ResMLP_12(1000)

    resmlp = ResMLPWithLinearClassifier(resmlp_config)
    print(resmlp)

    x = torch.randn(1, 3, 224, 224)

    print("Input Shape:\n", x.shape)
    print("Output Shape:\n", resmlp(x).shape)