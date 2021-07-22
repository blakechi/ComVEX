import os
import gc
import torch
from .utils import *

# === Import model-related objects ===


# === Instantiate your Model ===
# - For single model
model_name = os.path.basename(__file__).replace("test_", "").replace(".py", "")
model = None
# - For specializations
# specializations = [attr for attr in dir(__your_config_class_object__) if attr.startswith(model_name)]

# === Settings ===
# - Required:
input_shape = None
expected_shape = None
# - Optional:

# === Test Cases ===
# Default test for the single model case
def test_forward():
    model.eval()

    x = torch.randn(input_shape)
    out = model(x)

    assert_output_shape_wrong(out, expected_shape)
    assert_output_has_nan(out)

# Default test for specializations
# def test_forward():
#     for spec in specializations:
#         print(spec)
#         config = getattr(__your_config_class_object__, spec)(__possible_args__)
#         model = __your_class_object__(config)
#         model.eval()

#         x = torch.randn(input_shape)
#         out = model(x)

#         assert_output_shape_wrong(out, expected_shape)
#         assert_output_has_nan(out)
    
#         del model
#         gc.collect()

# Add other tests if any
def test_something():
    ...
