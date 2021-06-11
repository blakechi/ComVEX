# gMLP (Underconstruction)

This is an PyTorch implementation of [Pay Attention to MLPs](https://arxiv.org/abs/2105.08050). <!--  For the official implementation, check out this [repo](). -->

## Objects

1. `gMLPBase`
2. `SpatialGatingUnit`
3. `gMLPBlock`
4. `gMLPBackbone`
5. `gMLPWithLinearClassifier`
6. `gMLPConfig`
   - `gMLP_Ti`
   - `gMLP_S`
   - `gMLP_B`

## Usage

1. gMLP Configuration

```python
from comvex.g_mlp import gMLPConfig

g_mlp_config = gMLPConfig(
    image_channel=3,           # The number of image channels
    image_size=224,            # Image size
    patch_size=16,             # Patch size
    depth=30,                  # Number of layers
    ffn_dim=128,               # Token dimension
    num_classes=1000,          # The number of classes for classification
    pred_act_fnc_name="ReLU"   # The activation function for the projection head
    attention_dim=None,        # The dimension for the blended attention. `None` means don't use it.
    attention_dropout=0.0,     # Dropout rate for the attention maps
    token_dropout=0.0,         # Dropout rate for the tokens
    ff_dropout=0.1,            # Dropout rate for all feed forwarded networks
)
```

2. gMLP Backbone

```python
from comvex.g_mlp import gMLPBackbone

g_mlp_backbone = gMLPBackbone(
    3,                     # The number of image channels
    224,                   # Image size
    16,                    # Patch size
    depth=12,              # Number of layers
    token_mlp_dim=384,     # The dimension of the token mixer
    channel_mlp_dim=3072,  # The dimension of the channel mixer
    ff_dropout=0.1,        # The dropout rate for all feed forwarded networks
)
```

3. Specifications of the gMLP architectures

```python
from comvex.g_mlp import gMLPConfig, gMLPWithLinearClassifier

gmlp_config = gMLPConfig.gMLP_B()
gmlp_b = gMLPWithLinearClassifier(gmlp_config)
```

## Demo

```bash
python examples/gMLP/demo.py
```

## Citation

```bibtex
@misc{liu2021pay,
      title={Pay Attention to MLPs},
      author={Hanxiao Liu and Zihang Dai and David R. So and Quoc V. Le},
      year={2021},
      eprint={2105.08050},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
