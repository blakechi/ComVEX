import os
import sys

sys.path.insert(0, os.getcwd())

import torch 

from comvex.efficientdet import EfficientDetObjectDetectionConfig, EfficientDetObjectDetection


if __name__ == "__main__":

    efficientdet_config = EfficientDetObjectDetectionConfig.D0(10, 20)
    efficientdet = EfficientDetObjectDetection(efficientdet_config)

    x = torch.randn(1, 3, 512, 512)

    print("Input Shape:\n", x.shape)
    outs = efficientdet(x)
    for idx, out in enumerate(outs):
        print(f"Output Shape ({idx}):\n", out)
