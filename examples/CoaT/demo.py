import os
import sys

sys.path.insert(0, os.getcwd())

import torch 

from comvex.coat import CoaTLiteConfig, CoaTLiteWithLinearClassifier


if __name__ == "__main__":

    coat_config = CoaTLiteConfig.CoaTLite_Tiny(num_classes=10, attention_dropout=0.2)
    coat = CoaTLiteWithLinearClassifier(coat_config)

    print(coat)

    x = torch.randn(1, 3, 224, 224)

    print("Input Shape:\n", x.shape)
    print("Output Shape:\n", coat(x).shape)
