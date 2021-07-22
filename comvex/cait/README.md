# CaiT

This is an PyTorch implementation of [Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239). For the official implementation, check out this [repo](https://github.com/facebookresearch/deit).

## Objects

1. `LayerScale` (comvex/utils/layer_scale.py)
2. `ClassAttention`
3. `ClassAttentionLayer`
4. `SelfAttentionLayer`
5. `CaiTBackbone`
6. `CaiTWithLinearClassifie`
7. `CaiTConfig`
   - `CaiT_XXS_24`
   - `CaiT_XXS_36`
   - `CaiT_XS_24`
   - `CaiT_XS_36`
   - `CaiT_S_24``
   - `CaiT_S_36`
   - `CaiT_S_48`
   - `CaiT_M_24``
   - `CaiT_M_36`
   - `CaiT_M_48`

## Usage

1. CaiT Configuration

```python
from comvex.cait import CaiTWithLinearClassifier

cait = CaiTBacWithLinearClassifier(
    image_size = 224             # The laterral length of input images
    image_channel = 3            # Number of channels of input images
    patch_size = 16              # The lateral length of patches
    self_attn_depth = 24         # Number of layers for `SelfAttentionLayer`
    cls_attn_depth = 2           # Number of layers for `ClassAttentionLayer`
    dim = 192                    # The dimension of tokens
    alpha = 1e-5,                # The parameter for `LayerScale`. Normally a small number
    num_classes = 1000,          # Number of categories
    heads = None,                # Number of heads. Default: None (Since the official paper set the dimension of heads to 48)
    ff_expand_scale = 4,         # The expansion scale for the feed-forward blocks after the attention ones in each layer
    ff_dropout = 0.,             # The dropout rate for all feed-forward layers
    token_dropout = 0.           # The dropout rate for token dropout
    attention_dropout = 0.       # The dropout rate for attention maps
    path_dropout = 0.1,          # The dropout rate for stochastic depth
    pred_act_fnc_name = "ReLU",  # The activation function for the prediction head (choose one supported by PyTorch)
)
```

2. CaiT Backbone

```python
from comvex.cait import CaiTBackbone

cait_backbone = CaiTBackbone(
    image_size = 224        # The laterral length of input images
    image_channel = 3       # Number of channels of input images
    patch_size = 16         # The lateral length of patches
    self_attn_depth = 24    # Number of layers for `SelfAttentionLayer`
    cls_attn_depth = 2      # Number of layers for `ClassAttentionLayer`
    dim = 192               # The dimension of tokens
    alpha = 1e-5,           # The parameter for `LayerScale`. Normally a small number
    heads = None,           # Number of heads. Default: None (Since the official paper set the dimension of heads to 48)
    ff_expand_scale = 4,    # The expansion scale for the feed-forward blocks after the attention ones in each layer
    ff_dropout = 0.,        # The dropout rate for all feed-forward layers
    token_dropout = 0.      # The dropout rate for token dropout
    attention_dropout = 0.  # The dropout rate for attention maps
    path_dropout = 0.1      # The dropout rate for stochastic depth
)
```

3. Specifications of the CaiT architectures

```python
from comvex.cait import CaiTConfig, CaiTWithLinearClassifier

cait_config = CaiTConfig.CaiT_XXS_24(num_classes=1000)
cait = CaiTWithLinearClassifier(cait_config)
```

## Demo

```bash
python examples/CaiT/demo.py
```

## Citation

```bibtex
@misc{touvron2021going,
      title={Going deeper with Image Transformers},
      author={Hugo Touvron and Matthieu Cord and Alexandre Sablayrolles and Gabriel Synnaeve and Hervé Jégou},
      year={2021},
      eprint={2103.17239},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
