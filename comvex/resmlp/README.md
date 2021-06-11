# ResMLP

This is an PyTorch implementation of [ResMLP: Feedforward networks for image classification with data-efficient training](https://arxiv.org/abs/2105.03404). For the **related**-official implementation (Layer Scale from [CaiT](https://arxiv.org/abs/2103.17239)), check out this [repo](https://github.com/facebookresearch/deit/blob/main/cait_models.py#L130).

## Objects

1. `ResMLPLayer`
2. `ResMLPBackBone`
3. `ResMLPWithLinearClassifier`
4. `ResMLPConfig`
   - `ResMLP_12`
   - `ResMLP_24`
   - `ResMLP_36`

## Usage

1. ResMLP Configuration

```python
from comvex.resmlp import ResMLPConfig

resmlp_config = ResMLPConfig(
    image_size=224,             # Image size
    image_channel=3,            # Number of input image channels
    patch_size=16,              # Patch size
    depth=12,                   # Number of layers
    dim=384,                    # The token dimension
    num_classes=1000,           # Number of categories
    path_dropout=0.,            # Path dropout rate
    token_dropout=0.,           # Token dropout rate
    ff_dropout=0.,              # Feed forward layers' dropout rate
)
```

2. ResMLP Backbone

```python
from comvex.resmlp import ResMLPBackbone

resmlp_backbone = ResMLPBackbone(
    image_size=224,             # Image size
    image_channel=3,            # Number of input image channels
    patch_size=16,              # Patch size
    depth=12,                   # Number of layers
    dim=384,                    # The token dimension
    path_dropout=0.,            # Path dropout rate
    token_dropout=0.,           # Token dropout rate
    ff_dropout=0.,              # Feed forward layers' dropout rate
)
```

3. Specifications of the ResMLP architectures

```python
from comvex.resmlp import ResMLPConfig, ResMLPWithLinearClassifier

resmlp_config = ResMLPConfig.ResMLP_12()
resmlp = ResMLPWithLinearClassifier(resmlp_config)
```

## Demo

```bash
python examples/ResMLP/demo.py
```

## Citation

```bibtex
@misc{touvron2021resmlp,
      title={ResMLP: Feedforward networks for image classification with data-efficient training},
      author={Hugo Touvron and Piotr Bojanowski and Mathilde Caron and Matthieu Cord and Alaaeldin El-Nouby and Edouard Grave and Armand Joulin and Gabriel Synnaeve and Jakob Verbeek and Hervé Jégou},
      year={2021},
      eprint={2105.03404},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
