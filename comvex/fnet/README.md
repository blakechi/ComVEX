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
   - `FNet_B_8_512`
   - `FNet_Mini_4_512`
   - `FNet_Mini_4_256`
   - `FNet_Micro_2_256`
   - `FNet_Micro_2_128`

## Usage

1. FNet Configuration

```python
from comvex.fnet import FNetConfig

fnet_config = FNetConfig(
    image_size=224,             # Image size
    image_channel=3,            # Number of input image channels
    patch_size=16,              # Patch size
    dim=512,                    # The token dimension
    depth=12,                   # Number of encoder layers
    num_classes=1000,           # Number of the categories
    pre_norm=False,             # Whether to normalize before `fourier_block` and `ff_block`
    ff_dim=2048,                # Feed forward layers' expanding dimention
    ff_dropout=0.0,             # Feed forward layers' dropout rate
    token_dropout=0.0,          # Token dropout rate
    ff_act_fnc_name="ReLU",     # Activation function for the feed forward layers in the encoder layers
    dense_act_fnc_name="ReLU",  # Activation function for the `Dense` layer from the official paper
    pred_act_fnc_name="ReLU"    # Activation function for the projection head from the official paper
)
```

2. FNet Backbone

```python
from comvex.fnet import FNetBackbone

fnet_backbone = FNetBackbone(
    image_size=224,             # Image size
    image_channel=3,            # Number of input image channels
    patch_size=16,              # Patch size
    dim=512,                    # The token dimension
    depth=12,                   # Number of encoder layers
    pre_norm=False,             # Whether to normalize before `fourier_block` and `ff_block`
    ff_dim=2048,                # Feed forward layers' expanding dimention
    ff_dropout=0.0,             # Feed forward layers' dropout rate
    token_dropout=0.0,          # Token dropout rate
    ff_act_fnc_name="ReLU",     # Activation function for the feed forward layers in the encoder layers
    dense_act_fnc_name="ReLU",  # Activation function for the `Dense` layer from the official paper
)
```

3. Specifications of the FNet architectures

```python
from comvex.fnet import FNetConfig, FNetWithLinearClassifier

fnet_config = FNetConfig.FNet_B_12_512()
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
