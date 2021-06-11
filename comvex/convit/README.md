# ConViT

This is an PyTorch implementation of [ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases](https://arxiv.org/abs/2103.10697). For the official implementation, check out this [repo](https://github.com/facebookresearch/convit).

## Objects

1. `GatedPositionalSelfAttention`
2. `ConViTLayer`
3. `ConViTBackbone`
4. `ConViTWithLinearClassifier`
5. `ConViTConfig`
   - `ConViT_Ti`
   - `ConViT_Ti_plus`
   - `ConViT_S`
   - `ConViT_S_plus`
   - `ConViT_B`
   - `ConViT_B_plus`

## Usage

1. ConViT Configuration

```python
from comvex.convit import ConViTConfig

convit_config = ConViTConfig(
    image_size=224,             # Image size
    image_channel=3,            # Number of input image channels
    patch_size=16,              # Patch size
    num_local_layers=10,        # Number of layers before the adding the CLS token
    num_nonlocal_layers=2,      # Number of layers after the adding the CLS token
    dim=192,                    # The token dimension
    num_classes=1000,           # Number of categories
    locality_strength=1.,       # The `alpha` in the official paper
    heads=4,                    # Number of heads for attention. Choose `heads` or `head_dim` for your setting. Default: None
    head_dim=None,              # The dimention of each heads for attention. Choose `heads` or `head_dim` for your setting. Default: None
    pre_norm=False,             # Whether to normalize before `fourier_block` and `ff_block`
    ff_dim=2048,                # Feed forward layers' expanding dimention
    ff_dropout=0.,              # Feed forward layers' dropout rate
    attention_dropout=0.,       # Attention map's dropout rate (We use nn.Dropout2D)
    token_dropout=0.,           # Token dropout rate
    pred_act_fnc_name="ReLU"    # The name of the activation function for the projection head
)
```

2. ConViT Backbone

```python
from comvex.convit import ConViTBackbone

convit_backbone = ConViTBackbone(
    image_size=224,             # Image size
    image_channel=3,            # Number of input image channels
    patch_size=16,              # Patch size
    num_local_layers=10,        # Number of layers before the adding the CLS token
    num_nonlocal_layers=2,      # Number of layers after the adding the CLS token
    dim=192,                    # The token dimension
    locality_strength=1.,       # The `alpha` in the official paper
    heads=4,                    # Number of heads for attention. Choose `heads` or `head_dim` for your setting. Default: None
    head_dim=None,              # The dimention of each heads for attention. Choose `heads` or `head_dim` for your setting. Default: None
    pre_norm=False,             # Whether to normalize before `fourier_block` and `ff_block`
    ff_dim=2048,                # Feed forward layers' expanding dimention
    ff_dropout=0.,              # Feed forward layers' dropout rate
    attention_dropout=0.,       # Attention map's dropout rate (We use nn.Dropout2D)
    token_dropout=0.,           # Token dropout rate
)
```

3. Specifications of the ConViT architectures

```python
from comvex.convit import ConViTConfig, ConViTWithLinearClassifier

convit_config = ConViTConfig.ConViT_Ti()
convit = ConViTWithLinearClassifier(convit_config)
```

## Demo

```bash
python examples/ConViT/demo.py
```

## Citation

```bibtex
@misc{dascoli2021convit,
      title={ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases},
      author={Stéphane d'Ascoli and Hugo Touvron and Matthew Leavitt and Ari Morcos and Giulio Biroli and Levent Sagun},
      year={2021},
      eprint={2103.10697},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
