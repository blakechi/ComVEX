import torch
from .utils import *

# === Import model-related objects ===
from models.resmlp import ResMLPConfig, ResMLPWithLinearClassifier

# === Instantiate your Model ===
# - For specializations
specializations = [attr for attr in dir(ResMLPConfig) if attr.startswith("ResMLP")]

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
        config = getattr(ResMLPConfig, spec)(**kwargs)
        model = ResMLPWithLinearClassifier(config)
        model.eval()

        x = torch.randn(input_shape)
        out = model(x)

        assert_output_shape_wrong(out, expected_shape)
        assert_output_has_nan(out)
    
        del model