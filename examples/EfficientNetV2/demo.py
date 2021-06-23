import os
import sys

sys.path.insert(0, os.getcwd())

import torch 

from comvex.efficientnet_v2 import EfficientNetV2Config, EfficientNetV2WithLinearClassifier


if __name__ == "__main__":

    x = torch.randn(1, 3, 224, 224)

    # Scripting
    ## === if `return_feature_maps` == False
    # efficientnet_v2_config = EfficientNetV2Config.EfficientNetV2_S(num_classes=10, up_sampling_mode="bicubic", return_feature_maps=False)
    # efficientnet_v2 = torch.jit.script(EfficientNetV2WithLinearClassifier(efficientnet_v2_config))
    ## === else
    # efficientnet_v2_config = EfficientNetV2Config.EfficientNetV2_S(num_classes=10, up_sampling_mode="bicubic", return_feature_maps=True)
    # efficientnet_v2 = torch.jit.script(EfficientNetV2WithLinearClassifier(efficientnet_v2_config))

    # Tracing
    ## === if `return_feature_maps` == False
    # efficientnet_v2_config = EfficientNetV2Config.EfficientNetV2_S(num_classes=10, up_sampling_mode="bicubic", return_feature_maps=False)
    # efficientnet_v2 = torch.jit.trace(EfficientNetV2WithLinearClassifier(efficientnet_v2_config), x)
    ## === else
    efficientnet_v2_config = EfficientNetV2Config.EfficientNetV2_S(num_classes=10, up_sampling_mode="bicubic", return_feature_maps=True)
    efficientnet_v2 = torch.jit.trace(EfficientNetV2WithLinearClassifier(efficientnet_v2_config), x, strict=False)
    
    try:
        store_path = sys.argv[1]
        torch.jit.save(efficientnet_v2, os.path.join(store_path, 'traced_efficientnet_v2.pt'))
    except FileNotFoundError:
        print(f"The specified `store_path`: {store_path} doesn't exist")
    except:
        pass  # ignore errors like IndexError

    print(efficientnet_v2.code)

    print("Input Shape:\n", x.shape)
    # if `return_feature_maps` == False
    # print("Output Shape:\n", efficientnet_v2(x).shape)  
    # else
    print("Output Shape:\n", efficientnet_v2(x)['x'].shape)
