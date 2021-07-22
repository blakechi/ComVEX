# EfficientNetV2

This is an PyTorch implementation of [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298). For the official code, please check out [here](https://github.com/google/automl/tree/master/efficientnetv2). Support `torch.jit.trace` now. Progressive Learning is still under construction.

## Objects

1. `EfficientNetV2Base,`
2. `FusedMBConvXd,`
3. `EfficientNetV2Backbone,`
4. `EfficientNetV2WithLinearClassifier`
5. `EfficientNetV2BaseConfig,`
   - `EfficientNetV2_B`
   - `EfficientNetV2_S`
   - `EfficientNetV2_M`
   - `EfficientNetV2_L`
   - `EfficientNetV2_XL`
6. `EfficientNetV2Config`
   - `EfficientNetV2_S`
   - `EfficientNetV2_M`
   - `EfficientNetV2_L`
   - `EfficientNetV2_XL`
   - `EfficientNetV2_B0`
   - `EfficientNetV2_B1`
   - `EfficientNetV2_B2`
   - `EfficientNetV2_B3`

## Usage

1. EfficientNetV2 Configuration

```python
from comvex.efficientnet_v2 import EfficientNetV2Config, EfficientNetV2BaseConfig

efficientnet_v2_config = EfficientNetV2Config(
    base_config=EfficientNetV2BaseConfig.EfficientNetV2_S(),  # The configuration for the model architecture.
    image_channel=3,                                          # Number of image channels
    depth_scale=1.0,                                          # Depth scale
    width_scale=1.0,                                          # Width scale
    train_resolution=300,                                     # Image resolution when training
    eval_resolution=384,                                      # Image resolution when evaluating or testing
    num_classes=1000,                                         # Number of categories
    up_sampling_mode=True,                                    # If specified, upsample input images to `train_resolution` or `eval_resolution`. Check out the doc of `torch.nn.interpolate` for possible choices. Default: None (do nothing to the inputs)
    act_fnc_name="SiLU",                                      # The name of the activation function for all conv layers
    se_act_fnc_name="SiLU",                                   # The name of the activation function for `SEConv`
    se_scale=0.25,                                            # The dimension scale for `SEConv`
    batch_norm_eps=1e-3,                                      # Batch normalization's epsilon
    batch_norm_momentum=0.99,                                 # Batch normalization's momentum
    return_feature_maps=False,                                # Whether to return feature maps among stages
    path_dropout=0.2,                                         # The dropout rate for the path dropout. `path_dropout` == 1. - `survival_props` in the official paper
    ff_dropout=0.2                                            # The dropout rate for the only feed-forward layer in the output layer
)
```

2. EfficientNetV2 Backbone

```python
from comvex.efficientnet_v2 import EfficientNetV2Backbone, EfficientNetV2BaseConfig

efficientnet_v2_backbone = EfficientNetV2Backbone(
    base_config=EfficientNetV2BaseConfig.EfficientNetV2_S(),  # The configuration for the model architecture.
    image_channel=3,                                          # Number of image channels
    depth_scale=1.0,                                          # Depth scale
    width_scale=1.0,                                          # Width scale
    train_resolution=300,                                     # Image resolution when training
    eval_resolution=384,                                      # Image resolution when evaluating or testing
    up_sampling_mode=True,                                    # If specified, upsample input images to `train_resolution` or `eval_resolution`. Check out the doc of `torch.nn.interpolate` for possible choices. Default: None (do nothing to the inputs)
    act_fnc_name="SiLU",                                      # The name of the activation function for all conv layers
    se_act_fnc_name="SiLU",                                   # The name of the activation function for `SEConv`
    se_scale=0.25,                                            # The dimension scale for `SEConv`
    batch_norm_eps=1e-3,                                      # Batch normalization's epsilon
    batch_norm_momentum=0.99,                                 # Batch normalization's momentum
    return_feature_maps=False,                                # Whether to return feature maps among stages
    path_dropout=0.2,                                         # The dropout rate for the path dropout. `path_dropout` == 1. - `survival_props` in the official paper
)
```

3. Specifications of the EfficientNetV2 architectures

```python
from comvex.efficientnet_v2 import EfficientNetV2Config, EfficientNetV2WithLinearClassifier

efficientnet_v2_config = EfficientNetV2Config.EfficientNetV2_S(num_classes=10)
efficientnet_v2 = EfficientNetV2WithLinearClassifier(efficientnet_v2_config)
```

4. Tracing EfficientNetV2 (See its `demo.py`)

```python
import torch
from comvex.efficientnet_v2 import EfficientNetV2Config, EfficientNetV2WithLinearClassifier

efficientnet_v2_config = EfficientNetV2Config.EfficientNetV2_S(num_classes=10)
efficientnet_v2 = torch.jit.trace(EfficientNetV2WithLinearClassifier(efficientnet_v2_config))
```

## Demo

```bash
python examples/EfficientNetV2/demo.py
```

## Citation

```bibtex
@misc{tan2021efficientnetv2,
      title={EfficientNetV2: Smaller Models and Faster Training},
      author={Mingxing Tan and Quoc V. Le},
      year={2021},
      eprint={2104.00298},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
