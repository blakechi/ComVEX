# CoaT

This is an PyTorch implementation of [Co-Scale Conv-Attentional Image Transformers](https://arxiv.org/abs/2104.06399). For the official implementation, please check out [here](https://github.com/mlpc-ucsd/CoaT/blob/main/src/models/coat.py). Implemented `CoaTLite` only right now, will add `CoaT` together with `EfficientDet` and `DETR` for complete support for object detection and semantic segmentation tasks.

## Objects

1. `CoaTBase`
2. `FactorizedAttention`
3. `ConvAttentionalModule`
4. `CoaTSerialBlock`
5. `CoaTParallelBlock`
6. `CoaTLiteBackbone`
7. `CoaTLiteWithLinearClassifier`
8. `CoaTLiteConfig`
   - `CoaTLite_Tiny`
   - `CoaTLite_Mini`
   - `CoaTLite_Small`

## Usage

1. CoaT Configuration

```python
from comvex.coat import CoaTLiteConfig

coat_backbone = CoaTLiteConfig(
      image_channel=3,                            # The number of channel of input images
      image_size=224,                             # The lateral length of input images
      patch_size=4,                               # The lateral length of the patch size
      num_layers_in_stages=[2, 2, 2, 2],          # Number of layers in each stages
      num_channels=[64, 128, 256, 320],           # Number of channels in each stages
      expand_scales=[8, 8, 4, 4],                 # The expansion scale for `FactorizedAttention`'s feed-forward blocks in each stages
      heads=None,                                 # The number of heads for `FactorizedAttention`. Note that `heads` should be equal to the sum of the values of `kernel_size_on_heads`
      num_classes=1000,                           # The number of categories
      kernel_size_on_heads={ 3: 2, 5: 3, 7: 3 },  # kernel sizes for `ConvolutionalRelativePositionEncoding`. Keys are kernel sizes and values are the number of heads for each kernel size. Note the the sum of the heads in values should be equal to `heads` if specified.
      use_bias=True,                              # Whether to use bias in Q, K, V projection layers
      attention_dropout=0.1,                      # The dropout rate for attention maps
      ff_dropout=0.1,                             # The dropout rate for all feed-forward layers
      path_dropout=0.1,                           # The dropout rate for stocastic depth
      pred_act_fnc_name="ReLU"                    # The activation function name for the prediction head
)
```

2. CoaT Backbone

```python
from comvex.coat import CoaTLiteBackbone

coat_backbone = CoaTLiteBackbone(
      image_channel=3,                            # The number of channel of input images
      image_size=224,                             # The lateral length of input images
      patch_size=4,                               # The lateral length of the patch size
      num_layers_in_stages=[2, 2, 2, 2],          # Number of layers in each stages
      num_channels=[64, 128, 256, 320],           # Number of channels in each stages
      expand_scales=[8, 8, 4, 4],                 # The expansion scale for `FactorizedAttention`'s feed-forward blocks in each stages
      heads=None,                                 # The number of heads for `FactorizedAttention`. Note that `heads` should be equal to the sum of the values of `kernel_size_on_heads`
      kernel_size_on_heads={ 3: 2, 5: 3, 7: 3 },  # kernel sizes for `ConvolutionalRelativePositionEncoding`. Keys are kernel sizes and values are the number of heads for each kernel size. Note the the sum of the heads in values should be equal to `heads` if specified.
      use_bias=True,                              # Whether to use bias in Q, K, V projection layers
      attention_dropout=0.1,                      # The dropout rate for attention maps
      ff_dropout=0.1,                             # The dropout rate for all feed-forward layers
      path_dropout=0.1,                           # The dropout rate for stocastic depth
)
```

3. Specifications of the CoaTLite architectures

```python
from comvex.coat import CoaTLiteConfig, CoaTLiteWithLinearClassifier

coat_config = CoaTLiteConfig.CoaTLite_Tiny(num_classes=10, attention_dropout=0.2)
coat = CoaTLiteWithLinearClassifier(coat_config)
```

## Demo

```bash
python examples/CoaT/demo.py
```

## Citation

```bibtex
@misc{xu2021coscale,
      title={Co-Scale Conv-Attentional Image Transformers},
      author={Weijian Xu and Yifan Xu and Tyler Chang and Zhuowen Tu},
      year={2021},
      eprint={2104.06399},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
