# Vision Transformer (ViT)

This is an PyTorch implementation of [AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](https://arxiv.org/abs/2010.11929) referenced from [lucidrains](https://github.com/blakechi/vit-pytorch). For the official implementation, check out this [repo](https://github.com/google-research/vision_transformer).

Added new model configurations and enabled multihead attention pooling from [Scaling Vision Transformers](https://arxiv.org/abs/2106.04560). Removing CLS from patches and using multihead attention pooling is default now.

## Objects

1. `ViTBase`
2. `ViTBackbone`
3. `ViTWithLinearClassifier`
4. `ViTConfig`
   - `ViT_s_16`
   - `ViT_s_28`
   - `ViT_S_16`
   - `ViT_S_32`
   - `ViT_Ti_16`
   - `ViT_B_16`
   - `ViT_B_28`
   - `ViT_B_32`
   - `ViT_L_16`
   - `ViT_H_16`
   - `ViT_g_14`
   - `ViT_G_14`

## Usage

1. ViT Configuration

```python
from comvex.vit import ViTConfig

vit_config = ViTConfig(
    image_size=224,                          # Input image size
    image_channel=3,                         # Input image channel
    patch_size=16,                           # Patch size (one lateral of the square patch)
    dim=768,                                 # Token dimension
    depth=12,
    num_heads=12,
    num_classes=1000,                        # Number of categories
    use_multihead_attention_pooling = True,  # Whether to removing CLS from patches and using multihead attention pooling
    cat_cls_to_context = False,              # Whether to concatenate CLS to patches in multihead attention pooling. No operatoin when `use_multihead_attention_pooling` is False.
    pre_norm=False
    ff_dim=None,                             # If not specify, ff_dim = 4*dim
    ff_dropout=0.0,
    token_dropout=0.0,
    pred_act_fnc_name="ReLU"                 # The name of the activation function for the projection head
    self_defined_transformer=None,           # Use self-defined Transformer object
)
```

2. ViT Backbone

```python
from comvex.vit import ViTBackbone

vit_backbone = ViTBackbone(
    image_size=224,                          # Input image size
    image_channel=3,                         # Input image channel
    patch_size=16,                           # Patch size (one lateral of the square patch)
    dim=768,                                 # Token dimension
    depth=12,
    num_heads=12,
    use_multihead_attention_pooling = True,  # Whether to removing CLS from patches and using multihead attention pooling
    cat_cls_to_context = False,              # Whether to concatenate CLS to patches in multihead attention pooling. No operatoin when `use_multihead_attention_pooling` is False.
    pre_norm=False
    ff_dim=None,                             # If not specify, ff_dim = 4*dim
    ff_dropout=0.0,
    token_dropout=0.0,
    self_defined_transformer=None,           # Use self-defined Transformer object
)
```

3. Specifications of the ViT architectures

```python
from comvex.vit import ViTConfig, ViTWithLinearClassifier

vit_config = ViTConfig.ViT_B_16()
vit = ViTWithLinearClassifier(vit_config)
```

## Demo

```bash
python examples/ViT/demo.py
```

## Citation

```bibtex
@misc{dosovitskiy2020image,
      title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
      author={Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
      year={2020},
      eprint={2010.11929},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bibtex
@misc{zhai2021scaling,
      title={Scaling Vision Transformers},
      author={Xiaohua Zhai and Alexander Kolesnikov and Neil Houlsby and Lucas Beyer},
      year={2021},
      eprint={2106.04560},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
