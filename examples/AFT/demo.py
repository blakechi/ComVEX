import os
import sys

sys.path.insert(0, os.getcwd())

import torch 

from comvex.aft import AFTConfig, AFTWithLinearClassifier


if __name__ == "__main__":

    aft_config = AFTConfig.AFT_Conv_small_16_11(num_classes=10, ff_dropout=0.1)
    aft = AFTWithLinearClassifier(aft_config)

    print(aft)

    x = torch.randn(1, 3, 224, 224)

    print("Input Shape:\n", x.shape)
    print("Output Shape:\n", aft(x).shape)
