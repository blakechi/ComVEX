import os
import sys

sys.path.insert(0, os.getcwd())

import torch 

from comvex.xcit import XCiTConfig, XCiTWithLinearClassifier
 

if __name__ == "__main__":

    xcit_config = XCiTConfig.XCiT_N12_224_16(num_classes=1000)
    xcit = XCiTWithLinearClassifier(xcit_config)

    print(xcit)

    x = torch.randn(1, 3, 224, 224)

    print("Input Shape:\n", x.shape)
    print("Output Shape:\n", xcit(x).shape)
