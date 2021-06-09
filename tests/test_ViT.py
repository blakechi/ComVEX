import torch
from .utils import *

# === Import model-related objects ===
from models.vit import ViT

# === Instantiate your Model ===
# - For single model
model = ViT(
    image_size=224,
    image_channel=1,
    patch_size=16,
    num_classes=10,
    dim=512,
    depth=12,
    num_heads=16,
    ff_dropout=0.0,
    pre_norm=True
)

# === Settings ===
# - Required:
input_shape = (1, 1, 224, 224)
expected_shape = (1, 10)
# - Optional:

# === Test Cases ===
# Default test for the single model case
def test_forward():
    model.eval()

    x = torch.randn(input_shape)
    out = model(x)

    assert_output_shape_wrong(out, expected_shape)
    assert_output_has_nan(out)