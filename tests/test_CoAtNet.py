import gc
import torch
from .utils import *

# === Import model-related objects ===
from comvex.coatnet import CoAtNetConfig, CoAtNetWithLinearClassifier

# === Instantiate your Model ===
# - For specializations
specializations = [attr for attr in dir(CoAtNetConfig) if attr.startswith("CoAtNet")]
specializations = specializations[:2]  # Avoid too large specializations

# === Settings ===
# - Required:
input_shape = (1, 3, 224, 224)
expected_shape = (1, 10)
# - Optional:
input_shape_larger = (1, 3, 384, 384)
input_shape_rect = (1, 3, 280, 336)
kwargs = {}
kwargs['num_classes'] = 10

# === Test Cases ===
# Default test for specializations
def test_forward():
    for spec in specializations:
        print(spec)
        config = getattr(CoAtNetConfig, spec)(**kwargs)
        model = CoAtNetWithLinearClassifier(config)
        model.eval()

        x = torch.randn(input_shape)
        out = model(x)

        assert_output_shape_wrong(out, expected_shape)
        assert_output_has_nan(out)
    
        del model
        gc.collect()


# Test when the input size gets larger at the inference time
def test_larger_shape_at_inference():
    for spec in specializations:
        print(spec)
        config = getattr(CoAtNetConfig, spec)(**kwargs)
        model = CoAtNetWithLinearClassifier(config)
        model.eval()

        x = torch.randn(input_shape_larger)
        out = model(x)

        assert_output_shape_wrong(out, expected_shape)
        assert_output_has_nan(out)
    
        del model
        gc.collect()

# Test when the input size isn't square at the inference time
def test_rect_shape_at_inference():
    for spec in specializations:
        print(spec)
        config = getattr(CoAtNetConfig, spec)(**kwargs)
        model = CoAtNetWithLinearClassifier(config)
        model.eval()

        x = torch.randn(input_shape_rect)
        out = model(x)

        assert_output_shape_wrong(out, expected_shape)
        assert_output_has_nan(out)
    
        del model
        gc.collect()