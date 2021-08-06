import os
import sys

sys.path.insert(0, os.getcwd())

import torch 

from comvex.efficientdet import EfficientDetObjectDetectionConfig, EfficientDetObjectDetection


if __name__ == "__main__":

    efficientdet_config = EfficientDetObjectDetectionConfig.D0(10, 20)
    efficientdet = EfficientDetObjectDetection(efficientdet_config)

    x = torch.randn(1, 3, 512, 512)
    pred_class, pred_box = efficientdet(x)

    print("Input Shape:\n", x.shape)
    for idx, out in enumerate(pred_class):
        print(f"Class Output Shape ({idx}):\n", out.shape)

    for idx, out in enumerate(pred_box):
        print(f"Box Output Shape ({idx}):\n", out.shape)