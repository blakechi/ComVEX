# Vision Transformer (ViT)

This is an PyTorch implementation of [AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](https://arxiv.org/abs/2010.11929) referenced from [lucidrains](https://github.com/blakechi/vit-pytorch). For the official implementation, check out this [repo](https://github.com/google-research/vision_transformer).

## Objects

1. `ViTBase`
2. `ViTBackbone`
3. `ViTWithLinearClassifier`
4. `ViTConfig`
   - `ViT_B`
   - `ViT_L`
   - `ViT_H`

## Usage

1. ViT Configuration

```python
from comvex.vit import ViTConfig

vit_config = ViTConfig(
    image_size=224,                 # Input image size
    image_channel=3,                # Input image channel
    patch_size=16,                  # Patch size (one lateral of the square patch)
    dim=768,                        # Token dimension
    depth=12,
    num_heads=12,
    num_classes=1000,           # Number of categories
    pred_act_fnc_name="ReLU"    # The name of the activation function for the projection head
    pre_norm=False
    ff_dim=None,                    # If not specify, ff_dim = 4*dim
    ff_dropout=0.0,
    token_dropout=0.0,
    self_defined_transformer=None,  # Use self-defined Transformer object
)
```

2. ViT Backbone

```python
from comvex.vit import ViTBackbone

vit_backbone = ViTBackbone(
    image_size=224,                 # Input image size
    image_channel=3,                # Input image channel
    patch_size=16,                  # Patch size (one lateral of the square patch)
    dim=768,                        # Token dimension
    depth=12,
    num_heads=12,
    pre_norm=False
    ff_dim=None,                    # If not specify, ff_dim = 4*dim
    ff_dropout=0.0,
    token_dropout=0.0,
    self_defined_transformer=None,  # Use self-defined Transformer object
)
```

3. Specifications of the ViT architectures

```python
from comvex.vit import ViTConfig, ViTWithLinearClassifier

vit_config = ViTConfig.ViT_B()
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
