import torch
from torch import nn

from models.vit import ViTBase
from models.utils import Residual, MultiheadAttention
from .config import gMLPConfig


class SpatialGatingUnit(nn.Module):
    def __init__(self, dim, num_tokens, attention_dim=None, **kwargs):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.spatial_proj = nn.Conv1d(num_tokens, num_tokens, 1)
        nn.init.zeros_(self.spatial_proj.weight)
        nn.init.ones_(self.spatial_proj.bias)

        self.blend_with_attention = Residual(
            MultiheadAttention(dim=dim, heads=1, head_dim=attention_dim, **kwargs)
        ) if attention_dim is not None else nn.Identity()

    def forward(self, x):
        
        skip, gated = torch.chunk(x, chunks=2, dim=-1)

        gated = self.norm(gated)
        gated = self.spatial_proj(gated)
        gated = self.blend_with_attention(gated)

        return skip*gated


class gMLPBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_tokens, *, attention_dim=None, **kwargs):
        super().__init__()

        self.net = Residual(
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, ffn_dim),
                nn.GELU(),
                SpatialGatingUnit(ffn_dim, num_tokens, attention_dim, **kwargs),
                nn.Linear(ffn_dim // 2, dim)
            )
        )

    def forward(self, x):
        return self.net(x)


class gMLPBackbone(nn.Module):
    def __init__(
        self,  
        depth, 
        dim, 
        ffn_dim,
        num_tokens,
        *,
        attention_dim=None,
        **kwargs
    ):
        super().__init__()

        self.layers = nn.Sequential(*[
            gMLPBlock(
                dim,  
                ffn_dim,
                num_tokens,
                attention_dim=attention_dim,
                **kwargs
            ) for _ in range(depth)
        ])
        
    def forward(self, x):
        return self.layers(x)


class gMLPViTBase(ViTBase):
    def __init__(self, config: gMLPConfig = None) -> None:
        super().__init__(config.image_size, config.image_channel, config.patch_size)





# TODO
# class gMLPDeiTBase(DeiTBase):
#     pass