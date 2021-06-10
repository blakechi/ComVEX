import torch
from .utils import *

# === Import model-related objects ===
from comvex.botnet import BoTNetConfig, BoTNetWithLinearClassifier

# === Instantiate your Model ===
# - For specializations
specializations = [attr for attr in dir(BoTNetConfig) if attr.startswith("BoTNet")]
specializations.pop(specializations.index("BoTNet_50_ImageNet"))

# === Settings ===
# - Required:
input_shape = (1, 3, 1024, 1024)
expected_shape = (1, 10)
# - Optional:
kwargs = {}
kwargs['num_classes'] = 10

# === Test Cases ===
# Default test for specializations
def test_forward():
    for spec in specializations:
        print(spec)
        config = getattr(BoTNetConfig, spec)(**kwargs)
        model = BoTNetWithLinearClassifier(config)
        model.eval()

        x = torch.randn(input_shape)
        out = model(x)

        assert_output_shape_wrong(out, expected_shape)
        assert_output_has_nan(out)
    
        del model