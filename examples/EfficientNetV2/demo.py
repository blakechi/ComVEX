import os
import sys

sys.path.insert(0, os.getcwd())

import torch 

from comvex.efficientnet_v2 import EfficientNetV2Config, EfficientNetV2WithLinearClassifier


if __name__ == "__main__":

    efficientnet_v2_config = EfficientNetV2Config.EfficientNetV2_S(num_classes=10, up_sampling_mode="bicubic", return_feature_maps=False)
    efficientnet_v2 = EfficientNetV2WithLinearClassifier(efficientnet_v2_config)

    print(efficientnet_v2)

    x = torch.randn(1, 3, 224, 224)

    print("Input Shape:\n", x.shape)
    print("Output Shape:\n", efficientnet_v2(x).shape)
