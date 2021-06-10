import torch
from torch import nn
from einops import repeat

from comvex.vit import ViTBase
from comvex.utils import Residual, MultiheadAttention, ProjectionHead, TokenWiseDropout
from .config import gMLPConfig


class gMLPBase(ViTBase):
    def __init__(self, image_size, image_channel, patch_size) -> None:
        super().__init__(image_size, image_channel, patch_size)

        self.num_tokens = self.num_patches + 1  # Patches + CLS


class SpatialGatingUnit(nn.Module):
    def __init__(self, dim, num_tokens, attention_dim=None, **kwargs):
        super().__init__()

        self.norm = nn.LayerNorm(dim//2)
        self.spatial_proj = nn.Conv1d(num_tokens, num_tokens, 1)
        nn.init.zeros_(self.spatial_proj.weight)
        nn.init.ones_(self.spatial_proj.bias)

        self.attention = Residual(
            MultiheadAttention(dim//2, 1, proj_dim=attention_dim, **kwargs)
        ) if attention_dim is not None else None

    def forward(self, x):
        skip, gated = torch.chunk(x, chunks=2, dim=-1)

        gated = self.norm(gated)
        gated = self.spatial_proj(gated) + self.attention(gated) if self.attention is not None else self.spatial_proj(gated)

        return skip*gated


class gMLPBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_tokens, *, attention_dim=None, **kwargs):
        super().__init__()

        self.net = Residual(
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, ffn_dim),
                nn.GELU(),
                SpatialGatingUnit(
                    ffn_dim,
                    num_tokens,
                    attention_dim,
                    **kwargs
                ),
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


class gMLPWithLinearClassifier(gMLPBase):
    def __init__(self, config: gMLPConfig = None) -> None:
        super().__init__(config.image_size, config.image_channel, config.patch_size)

        self.linear_proj = nn.Linear(self.patch_dim, config.dim, bias=False)
        self.CLS = nn.Parameter(torch.randn(1, 1, config.dim), requires_grad=True)
        # self.token_dropout = nn.Dropout(config.token_dropout)
        self.token_dropout = TokenWiseDropout(config.token_dropout)

        self.backbone = gMLPBackbone(num_tokens=self.num_tokens, **config.__dict__)

        self.proj_head = ProjectionHead(
            config.dim,
            config.num_classes,
            config.pred_act_fnc_name,
        )

    def forward(self, x):
        b, _, _, _ = x.shape  # b, c, h, w = x.shape

        x = self.patch_and_flat(x)
        x = self.linear_proj(x)
        x = self.token_dropout(x)

        # Prepend CLS token and add position code
        CLS = repeat(self.CLS, "1 1 d -> b 1 d", b=b)
        x = torch.cat([CLS, x], dim=1)

        x = self.backbone(x)

        return self.proj_head(x[:, 0, :])
        