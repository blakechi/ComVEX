from math import log
import warnings

import torch
from torch import nn
from einops import rearrange, repeat

from comvex.utils import Residual, LayerNorm, FeedForward, MultiheadAttention


class PerceiverBlock(nn.Module):
    def __init__(
        self, 
        *, 
        cross_kv_dim, 
        cross_heads, 
        dim, 
        heads, 
        latent_transformer_depth=1, 
        pre_norm=False, 
        ff_dim_scale=4, 
        **kwargs
    ):
        super().__init__()

        if cross_heads > 1:
            warnings.warn(f"[{self.__class__.__name__}] `cross_heads` is set to {cross_heads}, but its 1 in the original paper.")

        self.cross_attention_block = nn.ModuleList([
            LayerNorm(
                dim=dim,
                cross_dim=cross_kv_dim,
                use_pre_norm=pre_norm,
                use_cross_attention=True,
                fn=Residual(
                    fn=MultiheadAttention(dim, kv_dim=cross_kv_dim, heads=cross_heads, **kwargs)
                )
            ),
            LayerNorm(
                dim=dim,
                use_pre_norm=pre_norm,
                fn=Residual(
                    fn=FeedForward(dim=dim, hidden_dim=ff_dim_scale*dim, **kwargs)
                )
            )
        ])

        self.latent_transformers = nn.ModuleList([
            nn.ModuleList([
                LayerNorm(
                    fn=Residual(
                        fn=MultiheadAttention(dim, heads=heads, **kwargs)
                    ),
                    dim=dim,
                    use_pre_norm=pre_norm,
                ),
                LayerNorm(
                    fn=Residual(
                        fn=FeedForward(dim=dim, hidden_dim=ff_dim_scale*dim, **kwargs)
                    ),
                    dim=dim,
                    use_pre_norm=pre_norm,
                )
            ]) for _ in range(latent_transformer_depth)
        ])
            
    def forward(self, latent, byte, attention_mask=None):
        """
        attention_mask doesn't exist in the official paper, but still put it here for future expansions.
        """

        cross_attn, cross_ff = self.cross_attention_block
        latent = cross_attn((latent, byte, byte), attention_mask)
        latent = cross_ff(latent)

        for latent_attn, latent_ff in self.latent_transformers:
            latent = latent_attn(latent)
            latent = latent_ff(latent)

        return latent


class Perceiver(nn.Module):
    def __init__(
        self, 
        *, 
        data_shape,  # Channel Major ex: image -> [C, H, W]
        cross_heads,
        num_latent_tokens,
        dim, 
        heads, 
        layers_indice, 
        num_latent_transformers_in_layers, 
        num_bands,
        resolution,
        frequency_base=2,
        pre_norm=False,
        ff_dim=None, 
        ff_dim_scale=4, 
        ff_dropout=0.0,
        attention_dropout=0.0,
        cross_kv_dim=None,
        head_dim=None
    ):
        super().__init__()

        self.num_bands = num_bands
        self.max_resolution = resolution / 2  # Nyquist frequency
        self.frequency_base = frequency_base
        self.pi = torch.acos(torch.zeros(1)).item() * 2

        num_unique_layers = len(set(layers_indice))
        assert num_unique_layers == len(num_latent_transformers_in_layers), (
            f"[{self.__class__.__name__}] The number of unique layers (Perceiver blocks) should be equal to the length of `num_latent_transformers_in_layers`."
        )
        self.layers_indice = layers_indice
        data_feature_dimension, num_data_axis_without_feature = data_shape[0], len(data_shape) - 1
        if cross_kv_dim is not None:
            assert cross_kv_dim == data_feature_dimension + num_data_axis_without_feature*(2*num_bands + 1), f"[{self.__class__.__name__}] `cross_kv_dim` should be {num_data_axis_without_feature*(2*num_bands + 1)}"
        else:
            cross_kv_dim = data_feature_dimension + num_data_axis_without_feature*(2*num_bands + 1)

        self.latent_array = nn.Parameter(torch.randn(num_latent_tokens, dim))
        torch.nn.init.xavier_normal_(self.latent_array)

        self.layers = nn.ModuleList([
            PerceiverBlock(
                cross_kv_dim=cross_kv_dim,
                cross_heads=cross_heads,
                dim=dim, 
                heads=heads, 
                latent_transformer_depth=latent_depth,
                pre_norm=pre_norm,
                ff_dim=ff_dim, 
                ff_dim_scale=ff_dim_scale,
                ff_dropout=ff_dropout,
                attention_dropout=attention_dropout,
                head_dim=head_dim
            )
            for latent_depth in num_latent_transformers_in_layers
        ])

    @staticmethod
    def get_position_code(axis, max_resolution, num_bands, frequency_base, pi, dtype, device="cpu"):
        # Adpated from https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_pytorch.py#L31

        normalized_grid = [torch.linspace(start=-1, end=1, steps=ax, device=device, dtype=dtype) for ax in axis]
        coordinate = torch.stack(torch.meshgrid(*normalized_grid), dim=-1)
        coordinate = coordinate.unsqueeze(dim=-1)  # To broadcast to num_bands when multiplying with freq at Line 156

        freq = torch.logspace(start=1, end=(log(max_resolution)/log(frequency_base)), steps=num_bands, base=frequency_base, dtype=dtype, device=device)
        freq = freq[[None for _ in range(len(coordinate.shape) - 1)] + [...]]  # Expand dimensions to (1, ..., 1, num_bands)
        freq = freq*pi*coordinate
        fourier_features = torch.cat([freq.sin(), freq.cos()], dim=-1)

        return torch.cat([coordinate, fourier_features], dim=-1)

    def forward(self, x, attention_mask=None):
        x = rearrange(x, "b d ... -> b ... d")
        b, *axis, d, dtype, device = *x.shape, x.dtype, x.device
        latent = repeat(self.latent_array, "l d -> b l d", b=b)

        # Position Encoding
        position = Perceiver.get_position_code(axis, self.max_resolution, self.num_bands, self.frequency_base, self.pi, dtype, device)
        position = repeat(position, "... n d -> b ... (n d)", b=b)

        # Concatenate Position Encoding
        x = torch.cat([x, position], dim=-1)

        # Flatten Inputs
        x = rearrange(x, "b ... d -> b (...) d")

        # Loop through each layer
        for idx in self.layers_indice:
            latent = self.layers[idx](latent, x, attention_mask)

        return latent.mean(dim=-2)