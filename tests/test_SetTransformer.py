import torch
from .utils import *

# === Import model-related objects ===
from comvex.set_transformer import SetTransformer, ISAB

# === Instantiate your Model ===
# - For single model
model = SetTransformer(
    dim=64, 
    heads=4, 
    encoder_base_block=ISAB,
    num_inducing_points=16, 
    num_seeds=4, 
    attention_dropout=0.0, 
    ff_dropout=0.0, 
    ff_expand_scale=4, 
    pre_norm=False, 
    head_dim=None
)

# === Settings ===
# - Required:
input_shape = (1, 4, 64)
expected_shape = (1, 4, 64)
# - Optional:

# === Test Cases ===
# Default test for the single model case
def test_forward():
    model.eval()

    x = torch.randn(input_shape)
    out = model(x)

    assert_output_shape_wrong(out, expected_shape)
    assert_output_has_nan(out)