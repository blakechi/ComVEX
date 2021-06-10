# Set Transformer

This is an implementation of the paper [Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks](https://arxiv.org/abs/1810.00825).

## Objects

1. `SAB`
2. `MAB`
3. `ISAB`
4. `PMA`
5. `SetTransformerEncoder`
6. `SetTransformerDecoder`
7. `SetTransformer`

## Usage

```python
from comvex.set_transformer import SetTransformer, ISAB

set_transformer = SetTransformer(
    dim=512,
    heads=4,
    encoder_base_block=ISAB,
    num_inducing_points=16,
    num_seeds=4,
    attention_dropout=0.0,
    ff_dropout=0.0,
    ff_dim_scale=4,
    pre_norm=False,
    head_dim=None
)
```

## Demo

```bash
python examples/Set_Transformer/demo.py
```

## Citation

```bibtex
@misc{lee2019set,
      title={Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks},
      author={Juho Lee and Yoonho Lee and Jungtaek Kim and Adam R. Kosiorek and Seungjin Choi and Yee Whye Teh},
      year={2019},
      eprint={1810.00825},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
