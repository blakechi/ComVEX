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

```

2. CoaT Backbone

```python
from comvex.coat import CoaTLiteBackbone

coat_backbone = CoaTLiteBackbone(
      image_channel=3,
      image_size=224,
      patch_size=4,
      num_layers_in_stages=[2, 2, 2, 2],
      num_channels=[64, 128, 256, 320],
      expand_scales=[8, 8, 4, 4],
      heads=None,
      kernel_size_on_heads={ 3: 2, 5: 3, 7: 3 },  #
      use_bias=True,
      attention_dropout=0.1,
      ff_dropout=0.1,
      path_dropout=0.1,
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
