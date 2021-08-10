# AFT

This is an PyTorch implementation of [An Attention Free Transformer](https://arxiv.org/abs/2105.14103).

## Objects

1. `AFTGeneral`
2. `AFTFull`
3. `AFTSimple`
4. `AFTLocal`
5. `AFTConv`
6. `AFTLayer`
7. `AFTBackbone`
8. `AFTWithLinearClassifier`
9. `AFTConfig`
   - `AFT_Full_tiny`
   - `AFT_Full_small`
   - `AFT_Conv_tiny_32_11`
   - `AFT_Conv_tiny_192_11`
   - `AFT_Conv_small_16_11`
   - `AFT_Conv_small_384_11`
   - `AFT_Conv_small_384_15`

## Usage

1. AFT Configuration

```python
from comvex.aft import AFTWithLinearClassifier

aft = AFTBacWithLinearClassifier(
    image_size=224,                # The laterral length of input images
    image_channel=3,               # Number of channels of input images
    patch_size=16,                 # The lateral length of patches
    num_layers=12,                 # Number of layers
    dim=192,                       # The dimension of tokens
    num_classes=1000,              # Number of categories for classification
    local_window_size=11,          # Local window's size. Default: 0. Set to None when `aft_mode` is either `full` or `simple` and to an integer greater than 0 when `local`.
    hidden_dim=None,               # The dimension after Q, K, V projectors. If is None, `hidden_dim` == `dim`.
    aft_mode="conv",               # AFT modes. Options: "general", "full", "simple", "local", or "conv".
    pool_mode="mean",              # The pooling mode for the output tokens. Options: "mean" or "class" (Class Attention Layer)
    query_act_fnc_name="Sigmoid",  # The query's activation function name. Reference torch.nn for options
    use_bias=True,                 # Whether to use biases on the Q, K, V projectors
    ff_expand_scale=4,             # The expanding scale for the feed-forward blocks in each layers
    ff_dropout=0.1,                # The dropout rate of linear layers
    attention_dropout=0.1,         # The dropout rate of attention maps
    path_dropout=0.1,              # The dropout rate of paths
    # AFT - General, Full, Simple, Local (arguments effect only when `aft_mode` is either "general", "full", "simple", or "local")
    position_bias_dim=128,         # The embedding dimension of position biases
    use_position_bias=True,        # Whehter to use position biases (Simple: False)
    # AFT - Conv (arguments effect only when `aft_mode` is "conv")
    heads=32,
    epsilon=1e-6,
    # Possible Class Attention Layer (arguments effect only when `pool_mode` is "class")
    alpha=1e-6,
    cls_attn_heads=6,
    # Projection Head
    pred_act_fnc_name="ReLU"
)
```

2. AFT Backbone

```python
from comvex.aft import AFTBackbone

aft_backbone = AFTBackbone(
    image_size=224,                # The laterral length of input images
    image_channel=3,               # Number of channels of input images
    patch_size=16,                 # The lateral length of patches
    num_layers=12,                 # Number of layers
    dim=192,                       # The dimension of tokens
    local_window_size=11,          # Local window's size. Default: 0. Set to None when `aft_mode` is either `full` or `simple` and to an integer greater than 0 when `local`.
    hidden_dim=None,               # The dimension after Q, K, V projectors. If is None, `hidden_dim` == `dim`.
    aft_mode="conv",               # AFT modes. Options: "general", "full", "simple", "local", or "conv".
    pool_mode="mean",              # The pooling mode for the output tokens. Options: "mean" or "class" (Class Attention Layer)
    query_act_fnc_name="Sigmoid",  # The query's activation function name. Reference torch.nn for options
    use_bias=True,                 # Whether to use biases on the Q, K, V projectors
    ff_expand_scale=4,             # The expanding scale for the feed-forward blocks in each layers
    ff_dropout=0.1,                # The dropout rate of linear layers
    attention_dropout=0.1,         # The dropout rate of attention maps
    path_dropout=0.1,              # The dropout rate of paths
    # AFT - General, Full, Simple, Local (arguments effect only when `aft_mode` is either "general", "full", "simple", or "local")
    position_bias_dim=128,         # The embedding dimension of position biases
    use_position_bias=True,        # Whehter to use position biases (Simple: False)
    # AFT - Conv (arguments effect only when `aft_mode` is "conv")
    heads=32,
    epsilon=1e-6,
    # Possible Class Attention Layer (arguments effect only when `pool_mode` is "class")
    alpha=1e-6,
    cls_attn_heads=6
)
```

3. Specifications of the AFT architectures

```python
from comvex.aft import AFTConfig, AFTWithLinearClassifier

aft_config = AFTConfig.AFT_Full_tiny(num_classes=1000)
aft = AFTWithLinearClassifier(aft_config)
```

## Demo

```bash
python examples/AFT/demo.py
```

## Citation

```bibtex
@misc{zhai2021attention,
      title={An Attention Free Transformer},
      author={Shuangfei Zhai and Walter Talbott and Nitish Srivastava and Chen Huang and Hanlin Goh and Ruixiang Zhang and Josh Susskind},
      year={2021},
      eprint={2105.14103},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
