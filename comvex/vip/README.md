# ViP

This is an PyTorch implementation of [Vision Permutator: A Permutable MLP-Like Architecture for Visual Recognition](https://arxiv.org/abs/2106.12368) with slight modifications. For the official implementation, check out this [repo](https://github.com/Andrew-Qibin/VisionPermutator).

## Objects

1. `PermuteMLP`
2. `Permutator`
3. `ViPDownSampler`
4. `ViPBackBone`
5. `ViPWithLinearClassifier`
6. `ViPConfig`
   - `ViP_Small_14`
   - `ViP_Small_7`
   - `ViP_Medium_7`
   - `ViP_Large_7`

## Modifications

1. We use channel-first format.
2. We add dropout on the `weight_proj`, which is not in the [official code](https://github.com/Andrew-Qibin/VisionPermutator/blob/main/models/vip.py#L51).

## Usage

1. ViP Configuration

```python
from comvex.vip import ViPConfig

vip_config = ViPConfig(
    image_channel=3,                          # The number of input images' chaanel
    image_size=224,                           # One lateral's size of a squre image
    patch_size=14,                            # One lateral's size of a squre patch
    layers_in_stages=[4, 3, 8, 3],            # Number of layers in each stage
    channels_in_stages=[384, 384, 384, 384],  # Channels in each stage
    num_classes=1000,                         # Number of categories for classification
    use_weighted=True,                        # Whether to use `Weighted Permute - MLP` or `Permute - MLP`
    use_bias=False,                           # Whether to use bias on all 1x1 Conv2D and linear layers
    ff_dropout=0.,                            # The dropout rate for all 1x1 Conv2D and linear layers
    path_dropout=0.1,                         # The dropout rate for path
)
```

2. ViP Backbone

```python
from comvex.vip import ViPBackbone

vip_backbone = ViPBackbone(
    image_channel=3,                          # The number of input images' chaanel
    image_size=224,                           # One lateral's size of a squre image
    patch_size=14,                            # One lateral's size of a squre patch
    layers_in_stages=[4, 3, 8, 3],            # Number of layers in each stage
    channels_in_stages=[384, 384, 384, 384],  # Channels in each stage
    use_weighted=True,                        # Whether to use `Weighted Permute - MLP` or `Permute - MLP`
    use_bias=False,                           # Whether to use bias on all 1x1 Conv2D and linear layers
    ff_dropout=0.,                            # The dropout rate for all 1x1 Conv2D and linear layers
    path_dropout=0.1,                         # The dropout rate for path
)
```

3. Specifications of ViP architectures

```python
from comvex.vip import ViPWithLinearClassifier, ViPConfig

vip_config = ViPConfig.ViP_Small_14(num_classes=10, ff_dropout=0.1)
vip = ViPWithLinearClassifier(vip_config)
```

## Demo

```bash
python examples/ViP/demo.py
```

## Citation

```bibtex
@misc{hou2021vision,
      title={Vision Permutator: A Permutable MLP-Like Architecture for Visual Recognition},
      author={Qibin Hou and Zihang Jiang and Li Yuan and Ming-Ming Cheng and Shuicheng Yan and Jiashi Feng},
      year={2021},
      eprint={2106.12368},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
