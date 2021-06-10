import gc
import torch
from .utils import *

# === Import model-related objects ===
from comvex.perceiver import Perceiver

# === Instantiate your Model ===
# - For single model
model = Perceiver(
    data_shape=[3, 224, 224],
    cross_heads=1,
    num_latent_tokens=1024,
    dim=512, 
    heads=16, 
    layers_indice=[0] + [1]*7, 
    num_latent_transformers_in_layers=[6]*2, 
    num_bands=64,
    resolution=224,
    frequency_base=2,
    pre_norm=True,
    ff_dim=None, 
    ff_dim_scale=4, 
    ff_dropout=0.0,
    attention_dropout=0.0,
    cross_kv_dim=None,
    head_dim=None
)

# === Settings ===
# - Required:
input_shape = (1, 3, 224, 224)
expected_shape = (1, 512)
# - Optional:

# === Test Cases ===
# Default test for the single model case
def test_forward():
    model.eval()

    x = torch.randn(input_shape)
    out = model(x)

    assert_output_shape_wrong(out, expected_shape)
    assert_output_has_nan(out)

del model
gc.collect()