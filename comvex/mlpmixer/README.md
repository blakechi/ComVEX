# MLP-Mixer

This is an PyTorch implementation of [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601) with slight modifications. For the official implementation, check out this [repo](https://github.com/google-research/vision_transformer).

## Objects

1. `MLPMixerLayer`
2. `MLPMixerBackBone`
3. `MLPMixerWithLinearClassifier`
4. `MLPMixerConfig`
   - `MLPMixer_S_32`
   - `MLPMixer_S_16`
   - `MLPMixer_B_32`
   - `MLPMixer_B_16`
   - `MLPMixer_L_32`
   - `MLPMixer_L_16``
   - `MLPMixer_H_14`

## Modifications

1. We split images into patches and flatten them first before transforming patches into tokens.
2. We use `nn.Linear` instead of `nn.Conv2d` for the "Pre-patched Fully-connected" layer.

## Usage

1. MLPMixer Configuration

```python
from comvex.mlpmixer import MLPMixerConfig

mlp_mixer_config = MLPMixerConfig(
    image_channel=3,       # The number of image channels
    image_size=224,        # Image size
    patch_size=16,         # Patch size
    depth=12,              # Number of layers
    token_mlp_dim=384,     # The dimension of the token mixer
    channel_mlp_dim=3072,  # The dimension of the channel mixer
    num_classes=1000,      # The number of classes for classification
    ff_dropout=0.1,        # The dropout rate for all feed forwarded networks
)
```

2. MLPMixer Backbone

```python
from comvex.mlpmixer import MLPMixerBackBone

mlp_mixer_backbone = MLPMixerBackBone(
    3,                     # The number of image channels
    224,                   # Image size
    16,                    # Patch size
    depth=12,              # Number of layers
    token_mlp_dim=384,     # The dimension of the token mixer
    channel_mlp_dim=3072,  # The dimension of the channel mixer
    ff_dropout=0.1,        # The dropout rate for all feed forwarded networks
)
```

3. Specifications of the Mixer architectures

```python
from comvex.mlpmixer import MLPMixer, MLPMixerConfig

mlp_mixer_config = MLPMixerConfig.MLPMixer_H_14()
mlp_mixer_h_14 = MLPMixer(mlp_mixer_config)
```

## Demo

```bash
python examples/MLPMixer/demo.py
```

## Citation

```bibtex
@misc{tolstikhin2021mlpmixer,
      title={MLP-Mixer: An all-MLP Architecture for Vision},
      author={Ilya Tolstikhin and Neil Houlsby and Alexander Kolesnikov and Lucas Beyer and Xiaohua Zhai and Thomas Unterthiner and Jessica Yung and Daniel Keysers and Jakob Uszkoreit and Mario Lucic and Alexey Dosovitskiy},
      year={2021},
      eprint={2105.01601},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
