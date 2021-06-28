import gc
import torch
from .utils import *

# === Import model-related objects ===
from comvex.xcit import XCiTConfig, XCiTWithLinearClassifier

# === Instantiate your Model ===
# - For specializations
specializations = [attr for attr in dir(XCiTConfig) if attr.startswith("XCiT")]
specializations = specializations[:4]  # Avoid too many specializations

# === Settings ===
# - Required:
input_shape = (1, 3, 224, 224)
expected_shape = (1, 10)
# - Optional:
kwargs = {}
kwargs['num_classes'] = 10

# === Test Cases ===
# Default test for specializations
def test_forward():
    for spec in specializations:
        print(spec)
        config = getattr(XCiTConfig, spec)(**kwargs)
        model = XCiTWithLinearClassifier(config)
        model.eval()

        x = torch.randn(input_shape)
        out = model(x)

        assert_output_shape_wrong(out, expected_shape)
        assert_output_has_nan(out)
    
        del model
        gc.collect()