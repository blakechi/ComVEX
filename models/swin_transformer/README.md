# Swin Transformer

This is an PyTorch implementation of [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030). For the official implementation, check out this [repo](https://github.com/microsoft/Swin-Transformer).

## Objects

1. `WindowAttentionBase`
2. `WindowAttention`
3. `ShiftWindowAttention`
4. `SwinTransformerBase`
5. `SwinTransformerLayer`
6. `SwinTransformerBackbone`
7. `SwinTransformerWithLinearClassifier`
8. `SwinTransformerConfig`

   - `SwinTransformer_T`
   - `SwinTransformer_S`
   - `SwinTransformer_B`
   - `SwinTransformer_L`

## Usage

1. Swin Transformer Configuration

```python
from models.swin_transformer import SwinTransformerConfig

swin_config = SwinTransformerConfig(
      image_channel,          # The number of input images' channels
      image_size,             # Image size (one lateral of the square image)
      patch_size,             # Patch size (one lateral of the square patch)
      num_channels,           # The number of features for tokens (denoted as C in the original paper)
      num_layers_in_stages,   # A four integers list. Indicating the number of Swin Transformer Block in each stage
      head_dim,               # The dimension for W-MSA and SW-MSA
      window_size,            # A tuple. The window size
      shifts,                 # The number of pixels to shift
      num_classes,            # The number of classes for classification
      use_absolute_position,  # Whether to add absolute postion encodings on the input tokens
      use_checkpoint,         # Whether to use checkpoint as proposed in Sparse Transformer
      use_pre_norm,           # Whether to use pre-layer normalization for W-MSA and SW-MSA
      ff_dim,                 # The expanding dimension of W-MSA's and SW-MSA's feed forward layers
      ff_dropout,             # Dropout rate for the feed forward layers
      attention_dropout,      # Dropout rate for attention maps
      token_dropout,          # Dropout rate for input tokens
)
```

2. Specifications of the Swin architectures

```python
from models.swin_transformer import SwinTransformerConfig, SwinTransformerWithLinearClassifier

swin_config = SwinTransformerConfig.SwinTransformer_B(
      image_channel=3,
      image_size=224,
      num_classes=1000,
      use_absolute_position=False,
      use_checkpoint=False,
)
swin_transformer = SwinTransformerWithLinearClassifier(swin_config)
```

## Demo

```bash
python examples/Swin_Transformer/demo.py
```

## Citation

```bibtex
@misc{liu2021swin,
      title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
      author={Ze Liu and Yutong Lin and Yue Cao and Han Hu and Yixuan Wei and Zheng Zhang and Stephen Lin and Baining Guo},
      year={2021},
      eprint={2103.14030},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
