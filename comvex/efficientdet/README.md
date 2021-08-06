# EfficientDet

This is an PyTorch implementation of [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070). For the official code, please check out [here](https://github.com/google/automl/tree/master/efficientdet). `BiFPN` supports 1/2/3D data. Still work for supporting segmentation task.

## Objects

1. `BiFPNConfig` (`comvex/utils/bifpn`)
2. `BiFPNResizeXd` (`comvex/utils/bifpn`)
3. `BiFPNNodeBase` (`comvex/utils/bifpn`)
4. `BiFPNIntermediateNode` (`comvex/utils/bifpn`)
5. `BiFPNOutputEndPoint` (`comvex/utils/bifpn`)
6. `BiFPNOutputNode` (`comvex/utils/bifpn`)
7. `BiFPNLayer` (`comvex/utils/bifpn`)
8. `BiFPN` (`comvex/utils/bifpn`)
9. `EfficientDetBackbone`
10. `EfficientDetPredictionHead`
11. `EfficientDetClassNet`
12. `EfficientDetBoxNet`
13. `EfficientDetObjectDetection`
14. `EfficientDetBackboneConfig`

- `D0`
- `D1`
- `D2`
- `D3`
- `D4`
- `D5`
- `D6`
- `D7`
- `D7x`

15. `EfficientDetObjectDetectionConfig`

- `D0`
- `D1`
- `D2`
- `D3`
- `D4`
- `D5`
- `D6`
- `D7`
- `D7x`

## Usage

1. EfficientDet Backbone Configuration

```python
from comvex.efficientdet import EfficientDetBackboneConfig
from comvex.utils import EfficientNetBackboneConfig

efficientdet_backbone_config = EfficientDetBackboneConfig(
    efficientnet_backbone_config=EfficientNetBackboneConfig.B0(resolution=512, strides=[1, *([2]*7), 1]),  # EfficientNet Backbone Configuration
    image_shapes=(512, 512),           # Input images' shape. A tuple.
    bifpn_num_layers=3,                # The number of BiFPN layers
    bifpn_channel=64,                  # The hidden channels in BiFPN
    dimension=2,                       # The spatial dimension of the input data. 2 for images
    upsample_mode="nearest",           # Upsampling mode for BiFPN. Options: "nearest", "linear", "bilinear", "bicubic", "trilinear"
    use_bias=True,                     # Whether to use bias in convolution layers
    use_conv_after_downsampling=True,  # Whether to add a convolution layer after downsampling
    norm_mode="fast_norm",             # The normalization method when fusing feature maps. Options: "fast_norm", "softmax", "channel_fast_norm", "channel_softmax"
    batch_norm_epsilon=1e-5,           # Batch Normalization's epsilon
    batch_norm_momentum=1e-1,          # Batch Normalization's momentum
    feature_map_indices=None           # Optional. A list of indices to indicate a group of feature maps (0 ~ number of stages - 1) that will be feed into BiFPN
)
```

2. EfficientDet Object Detection Configuration

```python
from comvex.efficientdet import EfficientDetBackboneConfig, EfficientDetObjectDetectionConfig

efficientdet_config = EfficientDetObjectDetectionConfig(
    efficientdet_backbone_config=EfficientDetBackboneConfig.D0(),  # EfficientDet Backbone Configuration
    num_pred_layers=3,                                             # Number of layers for Class and Box nets
    num_classes=10,                                                # Number of classes
    num_anchors=100,                                               # Number of anchors
    use_seperable_conv=True,                                       # Whether to use seperable convolution in Class and Box nets
    path_dropout=0.,                                               # The path dropout rate for Class and Box nets
)
```

3. Specifications of the EfficientDet architectures

```python
from comvex.efficientdet import EfficientDetObjectDetectionConfig, EfficientDetObjectDetection

efficientdet_config = EfficientDetObjectDetectionConfig.D0(
    num_classes=10,
    num_anchors=100,
)
efficientdet = EfficientDetObjectDetection(efficientdet_config)
```

## Demo

```bash
python examples/EfficientDet/demo.py
```

## Citation

```bibtex
@misc{tan2020efficientdet,
      title={EfficientDet: Scalable and Efficient Object Detection},
      author={Mingxing Tan and Ruoming Pang and Quoc V. Le},
      year={2020},
      eprint={1911.09070},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
