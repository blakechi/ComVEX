# Vision Transformer (ViT)

This is an PyTorch implementation of [AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](https://arxiv.org/abs/2010.11929) referenced from [lucidrains](https://github.com/blakechi/vit-pytorch). For the official implementation, check out this [repo](https://github.com/google-research/vision_transformer).

## Objects

1. `ViTBase`
2. `ViT`

## Usage

```python
from models.vit import ViT

vit = ViT(
    image_size=224,                 # Input image size
    image_channel=3,                # Input image channel
    patch_size=16,                  # Patch size (one lateral of the square patch)
    num_classes=2,
    dim=512,                        # Token dimension
    depth=12,
    num_heads=16,
    # Optional arguments
    pre_norm=True
    ff_dim=None,                    # If not specify, ff_dim = 4*dim
    ff_dropout=0.0,
    token_dropout=0.0,
    self_defined_transformer=None,  # Use self-defined Transformer object
)
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
