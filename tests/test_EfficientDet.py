import gc
import pytest
from itertools import accumulate

import torch

from .utils import *

# === Import model-related objects ===
from comvex.efficientdet import EfficientDetBackboneConfig, EfficientDetBackbone

# === Instantiate your Model ===
# - For specializations
specializations = [attr for attr in dir(EfficientDetBackboneConfig) if attr.startswith("D")]
specializations = specializations[:2]  # Avoid too large specializations

# === Settings ===
# - Required:
# - Optional:
kwargs = {}

# === Test Cases ===
# Default test for specializations
def test_forward():
    for spec in specializations:
        print(spec)
        config = getattr(EfficientDetBackboneConfig, spec)(**kwargs)
        image_shapes = config.image_shapes
        channel = config.bifpn_channel
        scale_in_stages = [8, 16, 32, 64, 128]
        expected_shapes = [(1, channel, image_shapes[0] // scale, image_shapes[1] // scale) for scale in scale_in_stages]

        model = EfficientDetBackbone(**config.__dict__)
        model.eval()

        x = torch.randn(1, 3, *(image_shapes))
        out = model(x)
        
        for feature_map, expected_shape in zip(out, expected_shapes):
            print(feature_map.shape, expected_shape)
            assert_output_shape_wrong(feature_map, expected_shape)
            assert_output_has_nan(feature_map)
    
        del model
        gc.collect()