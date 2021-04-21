# Perceiver

This is an implementation of the paper [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206) with slightly modifications. Adapt the method for Fourier feature positional encodings from [lucidrains](https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_pytorch.py#L31)

## Objects

1. `PerceiverBlock`
2. `Perceiver`

## Usage

```python
from models.perceiver import Perceiver

perceiver = Perceiver(
    data_shape=[1, 224, 224],                 # Channel Major, ex: image -> [C H W]
    cross_heads=1,                            # Number of heads for the cross attention block
    num_latent_tokens=1024,                   # Latent array: [num_latent_tokens, dim]
    dim=512,                                  #
    heads=16,                                 # Number of heads for latent transformers
    layers_indice=[0] + [1]*7,                # a list of indice for indicating which layer go next. Ex: [0, 0, 0, 1, 1, 1, 2, 2, 2] means 3 unique layers and each of them iterates 3 times.
    num_latent_transformers_in_layers=[6]*2,  # Number of latent transformers in one layer, its length should be as same as the number of unique layers.
    num_bands=64,                             # Number of bands for Fourier features
    resolution=224,                           # max_resolution = resolution / 2
    frequency_base=2,                         # Frequency base for Fourier features
    pre_norm=True,
    ff_dim=None,
    ff_dim_scale=4,
    ff_dropout=0.0,
    attention_dropout=0.0,
    cross_kv_dim=None,
    head_dim=None
)
```

## Demo

```bash
python examples/Perceiver/demo.py
```

## Citation

```bibtex
@misc{jaegle2021perceiver,
      title={Perceiver: General Perception with Iterative Attention},
      author={Andrew Jaegle and Felix Gimeno and Andrew Brock and Andrew Zisserman and Oriol Vinyals and Joao Carreira},
      year={2021},
      eprint={2103.03206},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
