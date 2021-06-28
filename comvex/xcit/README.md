# XCiT

This is an PyTorch implementation of [XCiT: Cross-Covariance Image Transformers](https://arxiv.org/abs/2106.09681). For the official implementation, check out this [repo](https://github.com/facebookresearch/xcit). The implementation can be traced, but it limits its flexibility to any shapes of input images. The solution provided here is interpolating input images to a pre-defined size (`image_size`), which can be enabled by setting `up_upsampling_mode` to one of [supported modes](https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html?highlight=upsample).

## Objects

1. `PositionEncodingFourier` (comvex/utils/position_encodings.py)
2. `PatchProjection`
3. `LocalPatchInteraction`
4. `CrossCovarianceAttention`
5. `XCiTLayer`
6. `XCiTBackbone`
7. `XCiTWithLinearClassifier`
8. `XCiTConfig`
   - `XCiT_N12_224_16`
   - `XCiT_N12_384_8`
   - `XCiT_T12_224_16`
   - `XCiT_T12_384_8`
   - `XCiT_T24_224_16`
   - `XCiT_T24_384_8`
   - `XCiT_S12_224_16`
   - `XCiT_S12_384_8`
   - `XCiT_S24_224_16`
   - `XCiT_S24_384_8`
   - `XCiT_M24_224_16`
   - `XCiT_M24_384_8`
   - `XCiT_L24_224_16`
   - `XCiT_L24_384_8`

## Usage

1. XCiT Configuration

```python
from comvex.xcit import XCiTConfig

xcit_config = XCiTConfig(
    image_size=224            # The lateral length of input images
    image_channel=3           # Number of input images' channels
    patch_size=16             # The lateral length of patches (in pixels)
    self_attn_depth=12        # Number of layers for XCiT Layer
    cls_attn_depth=2          # Number of layers for Class Attention Layer
    dim=128                   # Token dimensions
    heads=4                   # Number of heads
    num_classes=1000          # Number of categories
    alpha=1e-5                # Alpha for Layer Scale
    local_kernel_size=3       # Kernel size for Local Patch Interaction
    act_fnc_name="GELU"       # Activation function name
    use_bias=True             # Whether to use bias on Q, K, V projectoers in CrossCovarianceAttention
    ff_expand_scale=4         # Expansion scale for the hidden dimension of feed-forward layers
    ff_dropout=0.             # The dropout rate for all feed-forward layers
    attention_dropout=0.      # The dropout rate for attention maps
    path_dropout=0.1          # The dropout rate for stochastic depth
    token_dropout=0.          # The dropout rate for tokens
    upsampling_mode=None      # Whether to interpolate input images to pre-defined size (`image_size`)
    pred_act_fnc_name="ReLU"  # Activation function name for the linear classifier
)
```

2. XCiT Backbone

```python
from comvex.xcit import XCiTBackbone

xcit_backbone = XCiTBackbone(
    image_size=224        # The lateral length of input images
    image_channel=3       # Number of input images' channels
    patch_size=16         # The lateral length of patches (in pixels)
    self_attn_depth=12    # Number of layers for XCiT Layer
    cls_attn_depth=2      # Number of layers for Class Attention Layer
    dim=128               # Token dimensions
    heads=4               # Number of heads
    alpha=1e-5            # Alpha for Layer Scale
    local_kernel_size=3   # Kernel size for Local Patch Interaction
    act_fnc_name="GELU"   # Activation function name
    use_bias=True         # Whether to use bias on Q, K, V projectoers in CrossCovarianceAttention
    ff_expand_scale=4     # Expansion scale for the hidden dimension of feed-forward layers
    ff_dropout=0.         # The dropout rate for all feed-forward layers
    attention_dropout=0.  # The dropout rate for attention maps
    path_dropout=0.1      # The dropout rate for stochastic depth
    token_dropout=0.      # The dropout rate for tokens
    upsampling_mode=None  # Whether to interpolate input images to pre-defined size (`image_size`)
)
```

3. Specifications of the XCiT architectures

```python
from comvex.xcit import XCiTConfig, XCiTWithLinearClassifier

xcit_config = XCiTConfig.XCiT_N12_224_16(num_classes=1000)
xcit = XCiTWithLinearClassifier(xcit_config)
```

## Demo

```bash
python examples/XCiT/demo.py
```

## Citation

```bibtex
@misc{elnouby2021xcit,
      title={XCiT: Cross-Covariance Image Transformers},
      author={Alaaeldin El-Nouby and Hugo Touvron and Mathilde Caron and Piotr Bojanowski and Matthijs Douze and Armand Joulin and Ivan Laptev and Natalia Neverova and Gabriel Synnaeve and Jakob Verbeek and Hervé Jegou},
      year={2021},
      eprint={2106.09681},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```

```
