# FNet

This is an PyTorch implementation of [FNet: Mixing Tokens with Fourier Transforms](https://arxiv.org/abs/2105.03824). <!--  For the official implementation, check out this [repo](). -->

## Objects

1. `FNetFourierTransform`
2. `FNetEncoderLayer`
3. `FNetBackbone`
4. `FNetWithLinearClassifier`
5. `FNetConfig`
   - `FNet_L_24`
   - `FNet_B_12_768`
   - `FNet_B_12_512`

## Usage

1. FNet Configuration

```python
from comvex.fnet import gMLPConfig

fnet_config = gMLPConfig(
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

2. FNet Backbone

```python
from comvex.fnet import FNetBackbone

fnet_backbone = FNetBackbone(
    3,                     # The number of image channels
    224,                   # Image size
    16,                    # Patch size
    depth=12,              # Number of layers
    token_mlp_dim=384,     # The dimension of the token mixer
    channel_mlp_dim=3072,  # The dimension of the channel mixer
    ff_dropout=0.1,        # The dropout rate for all feed forwarded networks
)
```

3. Specifications of the FNet architectures

```python
from comvex.fnet import FNetConfig, FNetWithLinearClassifier

fnet_config = gMLPConfig.FNet_B_12_512()
fnet = FNetWithLinearClassifier(fnet_config)
```

## Demo

```bash
python examples/FNet/demo.py
```

## Citation

```bibtex
@misc{leethorp2021fnet,
      title={FNet: Mixing Tokens with Fourier Transforms},
      author={James Lee-Thorp and Joshua Ainslie and Ilya Eckstein and Santiago Ontanon},
      year={2021},
      eprint={2105.03824},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
