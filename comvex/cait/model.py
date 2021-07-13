from typing import Optional
from functools import partial
from collections import OrderedDict

import torch
from torch import nn
from einops import repeat

from comvex.vit import ViTBase
from comvex.utils import LayerScale, ClassMultiheadAttention, TalkingHeadAttention, MLP, TokenDropout, ProjectionHead
from comvex.utils.helpers.functions import config_pop_argument
from .config import CaiTConfig


ClassAttention = ClassMultiheadAttention
        

class ClassAttentionLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        alpha: float,
        ff_expand_scale: int = 4,
        ff_dropout: float = 0.,
        path_dropout: float = 0.,
        **kwargs
    ):
        super().__init__()

        self.attn_block = LayerScale(
            dim=dim,
            alpha=alpha,
            core_block=ClassAttention,
            ff_dropout=ff_dropout,
            path_dropout=path_dropout,
            cat_cls_to_context_at_dim=1,
            **kwargs
        )

        self.ff_block = LayerScale(
            dim=dim,
            alpha=alpha,
            core_block=MLP,
            expand_dim=dim*ff_expand_scale,
            ff_dropout=ff_dropout,
            path_dropout=path_dropout,
        )

    def forward(self, cls_token, x):
        out = self.attn_block(cls_token, x)
        out = self.ff_block(out)

        return out


class SelfAttentionLayer(nn.Module):
    r"""
    See: https://github.com/facebookresearch/deit/blob/main/cait_models.py#L130
    """
    def __init__(
        self,
        dim: int,
        alpha: float,
        ff_expand_scale: int = 4,
        ff_dropout: float = 0.,
        path_dropout: float = 0.,
        **kwargs
    ) -> None:
        super().__init__()

        self.attn_block = LayerScale(
            core_block=TalkingHeadAttention,
            dim=dim,
            alpha=alpha,
            ff_dropout=ff_dropout,
            path_dropout=path_dropout,
            **kwargs
        )

        self.ff_block = LayerScale(
            core_block=MLP,
            dim=dim,
            alpha=alpha,
            expand_dim=dim*ff_expand_scale,
            ff_dropout=ff_dropout,
            path_dropout=path_dropout,
        )

    def forward(self, x):
        x = self.attn_block(x)
        x = self.ff_block(x)

        return x


class CaiTBackbone(ViTBase):
    def __init__(
        self,
        image_size: int,
        image_channel: int,
        patch_size: int,
        self_attn_depth: int,
        cls_attn_depth: int,
        dim: int,
        alpha: float,
        heads: Optional[int] = None,
        ff_expand_scale: int = 4,
        ff_dropout: float = 0.,
        token_dropout: float = 0.,
        attention_dropout: float = 0.,
        path_dropout: float = 0.,
    ) -> None:
        super().__init__(image_size, image_channel, patch_size)

        heads = heads or dim // 48

        self.linear_proj = nn.Linear(self.patch_dim, dim, bias=False)
        self.CLS = nn.Parameter(torch.randn(1, 1, dim))
        self.position_code = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.token_dropout = TokenDropout(token_dropout)

        self.self_attn_layers = nn.Sequential(OrderedDict([
            (
                f"self_attn_layer_{idx}",
                SelfAttentionLayer(
                    dim=dim,
                    heads=heads,
                    alpha=alpha,
                    ff_expand_scale=ff_expand_scale,
                    ff_dropout=ff_dropout,
                    path_dropout=path_dropout,
                    attention_dropout=attention_dropout,
                )
            ) for idx in range(self_attn_depth)
        ]))

        self.cls_attn_layers = nn.ModuleList([
            ClassAttentionLayer(
                dim=dim,
                heads=heads,
                alpha=alpha,
                ff_expand_scale=ff_expand_scale,
                ff_dropout=ff_dropout,
                path_dropout=path_dropout,
                attention_dropout=attention_dropout,
            ) for _ in range(cls_attn_depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]  # b, c, h, w = x.shape 

        # Divide into flattened patches
        x = self.patch_and_flat(x)

        # Linear projection
        x = self.linear_proj(x)

        # Token dropout
        x = self.token_dropout(x)

        # Expand CLS token ann add position code
        cls_token = repeat(self.CLS, "1 1 d -> b 1 d", b=b)
        x = x + self.position_code
        
        # Self-Attention Layers
        x = self.self_attn_layers(x)
        
        # Classe Attention Layers
        for cls_layer in self.cls_attn_layers:
            cls_token = cls_layer(cls_token, x)

        return cls_token


class CaiTWithLinearClassifier(CaiTBackbone):
    def __init__(self, config: CaiTConfig) -> None:
        num_classes = config_pop_argument(config, "num_classes")
        pred_act_fnc_name = config_pop_argument(config, "pred_act_fnc_name")
        super().__init__(**config.__dict__)

        self.proj_head = ProjectionHead(
            dim=config.dim,
            out_dim=num_classes,
            act_fnc_name=pred_act_fnc_name
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        
        cls_token = super().forward(x).view(b, -1)

        return self.proj_head(cls_token)