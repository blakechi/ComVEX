# BoTNet

This is an implementation of the paper [Bottleneck Transformers for Visual Recognition](https://arxiv.org/abs/2101.11605).

## Objects

1. `BoTNetMHSA`
2. `BoTNetBlock`
3. `BoTNetFullPreActivationBlock`
4. `BoTNetBackBone`
5. `BoTNetWithLinearClassifier`
6. `BoTNetConfig`
   - `BoTNet_50`
   - `BoTNet_50_ImageNet`

## Usage

1. BoTNet Configuration

```python
from comvex.botnet import BoTNetConfig

botnet_config = BoTNetConfig(
    num_input_channel = 3,                                            # Number of channels of the input data
    input_lateral_size = 1024,                                        # The width or height of the input data. Assume the input is a squre matrix.
    conv_base_block_name = "ResNetFullPreActivationBottleneckBlock",  # The Conv block you want. Use BoTNetConfig.available_conv_base_blocks() to check.
    bot_base_block_name = "BoTNetFullPreActivationBlock",             # The BoT block you want. Use BoTNetConfig.available_bot_base_blocks() to check.
    num_blocks_in_layer = [3, 4, 6, 3],                               # A list of number of blocks in each layer for BoTNet. Same as ResNet.
    num_heads = 4,                                                    # Number of heads for multihead self-attention
    bot_block_indicator = [1, 1, 1],                                  # Indicator for whether to use the BoT block. 1 means BoT block and 0 means Conv block as specified in "conv_base_block_name". Note that the length of "bot_block_indicator" should be equal to the number of blocks in the last layer, which means len(bot_block_indicator) == num_blocks_in_layer[-1]
    num_classes = 1000,                                               # Optional, number of classes for classification tasks. Needed when using "BoTNetWithLinearClassifier"
)
```

2. BoTNet 50 Back Bone

```python
from comvex.botnet import BoTNetBackBone, BoTNetConfig

botnet_50_config = BoTNetConfig.BoTNet_50()
botnet_50 = BoTNetBackBone(botnet_50_config)
```

3. BoTNet 50 with Linear Classifier

```python
from comvex.botnet import BoTNetWithLinearClassifier, BoTNetConfig

botnet_50_config = BoTNetConfig.BoTNet_50_ImageNet()
botnet_50 = BoTNetWithLinearClassifier(botnet_50_config)
```

## Demo

```bash
python examples/BoTNet/demo.py
```

## Citation

```bibtex
@misc{srinivas2021bottleneck,
      title={Bottleneck Transformers for Visual Recognition},
      author={Aravind Srinivas and Tsung-Yi Lin and Niki Parmar and Jonathon Shlens and Pieter Abbeel and Ashish Vaswani},
      year={2021},
      eprint={2101.11605},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
