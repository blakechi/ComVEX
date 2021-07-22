# CoAtNet

This is an PyTorch implementation of [CoAtNet: Marrying Convolution and Attention for All Data Sizes](https://arxiv.org/abs/2106.04803). Support dynamic bilinear interpolation for relative biases when the height and width of hidden tensors don't align with the ones used for initializing CoAtNet. Users can interpolate relative biases and store them statically in inference for different sizes of images by using `update_relative_bias_and_indices` in `CoAtNetRelativeAttention`.

## Objects

1. `CoAtNetBase`
2. `CoAtNetRelativeAttention`
3. `CoAtNetTransformerBlock`
4. `CoAtNetConvBlock`
5. `CoAtNetBackbone`
6. `CoAtNetWithLinearClassifier`
7. `CoAtNetConfig`
   - `CoAtNet_0`
   - `CoAtNet_1`
   - `CoAtNet_2`
   - `CoAtNet_3`
   - `CoAtNet_4`

## Usage

1. CoAtNet Configuration

```python
from comvex.coatnet import CoAtNetConfig

coatnet_config = CoAtNetConfig(
    image_height=224,                                # Image height
    image_width=224,                                 # Image width
    image_channel=3,                                 # Image channel
    num_blocks_in_layers=[2, 2, 3, 5, 2],            # Number of block in each layer. We consider the first block as well, so its length must be 5.
    block_type_in_layers=['C', 'C', 'T', 'T'],       # The block type for each layer. "C" for MBConv and "T" for relative Transformer as specified in the offical paper. Its length must be 4
    num_channels_in_layers=[64, 96, 192, 384, 768],  # Number of transformed channels in each layer. The length of `num_channels_in_layers` must be 5.
    expand_scale_in_layers=4,                        # The scale of channel expansion in each layer. If given a integar, all layers would have the same scale; on the other hand, if given a list, the length of the list must be 4.
    heads=32,                                        # Number of heads for relative Transformer
    num_classes=1000,                                # Number of categories
    pred_act_func_name="ReLU",                       # Projection head's activation function. Check out PyTorch' doc for possible choices.
    ff_dropout=0.1,                                  # Dropout rate for all linear layers
    attention_dropout=0.1,                           # Dropout rate for the attention maps
    path_dropout=0.1,                                # Dropout rate for all residual connections
)
```

2. CoAtNet Backbone

```python
from comvex.coatnet import CoAtNetBackbone

coatnet_backbone = CoAtNetBackbone(
    image_height=224,                                # Image height
    image_width=224,                                 # Image width
    image_channel=3,                                 # Image channel
    num_blocks_in_layers=[2, 2, 3, 5, 2],            # Number of block in each layer. We consider the first block as well, so its length must be 5.
    block_type_in_layers=['C', 'C', 'T', 'T'],       # The block type for each layer. "C" for MBConv and "T" for relative Transformer as specified in the offical paper. Its length must be 4
    num_channels_in_layers=[64, 96, 192, 384, 768],  # Number of transformed channels in each layer. The length of `num_channels_in_layers` must be 5.
    expand_scale_in_layers=4,                        # The scale of channel expansion in each layer. If given a integar, all layers would have the same scale; on the other hand, if given a list, the length of the list must be 4.
    heads=32,                                        # Number of heads for relative Transformer
    ff_dropout=0.1,                                  # Dropout rate for all linear layers
    attention_dropout=0.1,                           # Dropout rate for the attention maps
    path_dropout=0.1,                                # Dropout rate for all residual connections
)
```

3. Specifications of the CoAtNet architectures

```python
from comvex.coatnet import CoAtNet, CoAtNetWithLinearClassifier

coatnet_config = CoAtNet.CoAtNet_0()
coatnet = CoAtNetWithLinearClassifier(coatnet_config)
```

## Demo

```bash
python examples/CoAtNet/demo.py
```

## Citation

```bibtex
@misc{dai2021coatnet,
      title={CoAtNet: Marrying Convolution and Attention for All Data Sizes},
      author={Zihang Dai and Hanxiao Liu and Quoc V. Le and Mingxing Tan},
      year={2021},
      eprint={2106.04803},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
