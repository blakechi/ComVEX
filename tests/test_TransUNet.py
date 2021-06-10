import torch
from .utils import *

# === Import model-related objects ===
from comvex.transunet import TransUNet

# === Instantiate your Model ===
# - For single model
model = TransUNet(
    input_channel=3,
    middle_channel=512,
    output_channel=2,
    channel_in_between=[64, 128, 256],
    num_res_blocks_in_between=[3, 4, 9],
    image_size=224,
    patch_size=2,
    dim=512,
    num_heads=16,
    num_layers=12,
    token_dropout=0,
    ff_dropout=0,
    to_remain_size=True
)

# === Settings ===
# - Required:
input_shape = (1, 3, 224, 224)
expected_shape = (1, 2, 224, 224)
# - Optional:

# === Test Cases ===
# Default test for the single model case
def test_forward():
    model.eval()

    x = torch.randn(input_shape)
    out = model(x)

    assert_output_shape_wrong(out, expected_shape)
    assert_output_has_nan(out)