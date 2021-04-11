import warnings

import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from einops.torch import Rearrange

from models.utils import Residual, LayerNorm, FeedForward, MultiheadAttention


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
            warnings.warn(f"[{self.__class__.__name__}] `cross_heads` is set to {cross_kv_dim}, but its 1 in the original paper.")

        self.cross_attention_block = nn.ModuleList([
            LayerNorm(
                dim=dim,
                use_pre_norm=pre_norm,
                fn=Residual(
                    fn=MultiheadAttention(dim=dim, kv_dim=cross_kv_dim, heads=cross_heads, **kwargs)
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
                    dim=dim,
                    use_pre_norm=pre_norm,
                    fn=Residual(
                        fn=MultiheadAttention(dim=dim, heads=heads, **kwargs)
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
            for _ in range(latent_transformer_depth)
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
        cross_kv_dim,
        cross_head,
        num_latent_tokens,
        dim, 
        heads, 
        layers_indice, 
        num_latent_transformers_in_layers, 
        pre_norm=False,
        ff_dim=None, 
        ff_dim_scale=4, 
        ff_dropout=0.0,
        head_dim=None
    ):
        super().__init__()

        num_unique_layers = len(set(layers_indice))
        assert num_unique_layers == len(num_latent_transformers_in_layers), (
            f"[{self.__class__.__name__}] The number of unique layers (Perceiver blocks) should be equal to the length of `num_latent_transformers_in_layers`."
        )
        self.layers_indice = layers_indice

        self.latent_array = nn.Parameter(torch.randn(num_latent_tokens, dim))
        torch.nn.init.xavier_normal_(self.latent_array)

        self.fourier_position_encoding

        self.layers = nn.ModuleList([
            PerceiverBlock(
                cross_kv_dim=cross_kv_dim,
                cross_head=cross_head,
                dim=dim, 
                heads=heads, 
                latent_transformer_depth=latent_depth,
                pre_norm=pre_norm,
                ff_dim=ff_dim, 
                ff_dim_scale=ff_dim_scale,
                ff_dropout=ff_dropout,
                head_dim=head_dim
            )
            for latent_depth in num_latent_transformers_in_layers
        ])

    def get_position_code(self, axis):
        ...

    def forward(self, x, attention_mask=None):
        b, *axis, d, device = *x.shape, x.device
        latent = repeat(self.latent_array, "l d -> b l d", b=b)

        # Flatten Inputs
        x = rearrange(x, "b ... d -> b (...) d")

        # Position Encoding
        
        # Concatenate Position Encoding
        x = rearrange([x, position_code], "c b n d -> b n (c d)")

        # Loop through each layer
        for idx in self.layers_indice:
            latent = self.layers[idx](latent, x, attention_mask)

        return latent.mean(dim=-2)