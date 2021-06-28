from sys import path
from typing import Optional, Tuple
from collections import OrderedDict
from math import log2

import torch
from torch import nn, einsum
from einops import rearrange, repeat
from torch.nn.modules.activation import GELU

from comvex.vit import ViTBase
from comvex.cait import ClassAttentionLayer
from comvex.utils import FeedForward, TokenDropout, ProjectionHead, LayerScale
from comvex.utils.helpers.functions import config_pop_argument, name_with_msg
from .config import XCiConfig


class PatchProjection(nn.Module):
    r"""
    Reference from: https://github.com/facebookresearch/xcit/blob/master/xcit.py#L68
    """
    def __init__(
        self,
        image_channel: int,
        patch_size: int,
        dim: int,
    ) -> None:
        super().__init__()
        
        assert (
            log2(patch_size) == int(log2(patch_size))
        ), name_with_msg(f"`patch_size: {patch_size} can't be divided by 2") 

        base_dimension_scale = 1 / (patch_size // 2)
        num_layers = int(log2(patch_size))

        self.proj = nn.Sequential(
            *[
                nn.Conv2d(
                    int(dim * base_dimension_scale * 2**((idx // 2) - 1)) if idx != 0 else image_channel, 
                    int(dim * base_dimension_scale * 2**(idx // 2)),
                    kernel_size=3,
                    stride=2,
                    padding=1
                ) if idx % 2 == 0 else nn.GELU() for idx in range(num_layers*2 - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class CrossCovarianceAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        attention_dropout: float = 0.,
        ff_dropout: float = 0.,
        use_bias: bool = False,
    ) -> None:
        super().__init__()

        assert (
            heads is not None or head_dim is not None
        ), f"[{self.__class__.__name__}] Either `heads` or `head_dim` must be specified."

        self.heads = heads if heads is not None else dim // head_dim
        head_dim = head_dim if head_dim is not None else dim // heads

        assert (
            head_dim * self.heads == dim
        ), f"[{self.__class__.__name__}] Head dimension times the number of heads must be equal to embedding dimension (`in_dim` or `proj_dim`)"
        
        self.QKV = nn.Linear(dim, dim*3, bias=use_bias)
        self.temperature = nn.Parameter(torch.ones(1, self.heads, 1, 1))

        self.out_linear = nn.Linear(dim, dim)

        self.attention_dropout = nn.Dropout2d(attention_dropout)
        self.out_dropout = nn.Dropout(ff_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H = x.shape[2]  # b, c, H, W = x.shape

        x = rearrange(x, "b c h w -> b (h w) c")  # h for height
        q, k, v = self.QKV(x).chunk(chunks=3, dim=-1)

        q = rearrange(q, "b n (h d) -> b h n d")  # h for heads
        k = rearrange(k, "b n (h d) -> b h d n")  # h for heads
        v = rearrange(v, "b n (h d) -> b h n d")  # h for heads

        q = torch.nn.functional.normalize(q, dim=-2)
        k = torch.nn.functional.normalize(k, dim=-1)*self.temperature
        attention = einsum("b h p n -> b h n q, b h p q", k, q)  # p == q == d
        attention = attention.softmax(dim=-1)
        attention = self.attention_dropout(attention)
        
        out = einsum("b h n p, b h p q -> b h n q", v, attention)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.out_dropout(self.out_linear(out))
        out = rearrange(out, "b (h w) c -> b c h w", h=H)

        return out


class LocalPatchInteraction(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: Optional[int] = None,
        kernel_size: int = 3,
        act_fnc_name: str = "GELU",
        epsilon: float = 1e-5,
        momentum: float = 1e-1,
        affine: bool = True,
    ) -> None:
        super().__init__()

        out_channel = out_channel or in_channel

        self.depth_wise_conv_0 = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=out_channel
        )
        self.depth_wise_conv_1 = nn.Conv2d(
            out_channel,
            out_channel,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=out_channel
        )

        self.act_fnc = getattr(nn, act_fnc_name)(),
        self.norm = nn.BatchNorm2d(out_channel, eps=epsilon, momentum=momentum, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_wise_conv_0(x)
        x = self.act_fnc(x)
        x = self.norm(x)
        x = self.depth_wise_conv_1(x)

        return x


class XCiTLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        act_fnc_name: str = "GELU",
        heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        use_bias: bool = False,
        ff_expand_scale: int = 4,
        ff_dropout: float = 0.,
        attention_dropout: float = 0.,
        path_dropout: float = 0.,
        alpha: float = 1e-5,
    ) -> None:
        super().__init__()

        self.lpi = LayerScale(
            dim,
            alpha=alpha,
            path_dropout=path_dropout,
            core_block=LocalPatchInteraction,
            kernel_size=kernel_size,
            act_fnc_name=act_fnc_name
        )
        self.xca = LayerScale(
            dim,
            alpha=alpha,
            path_dropout=path_dropout,
            core_block=CrossCovarianceAttention,
            heads=heads,
            head_dim=head_dim,
            use_bias=use_bias,
            attention_dropout=attention_dropout,
            ff_dropout=ff_dropout
        )
        self.ff = LayerScale(
            dim,
            alpha=alpha,
            path_dropout=path_dropout,
            core_block=FeedForward,
            ff_expand_scale=ff_expand_scale,
            ff_dropout=ff_dropout,
            use_convXd=2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.lpi(x)
        self.xca(x)
        self.ff(x)

        return x


class XCiTBackbone(ViTBase):
    def __init__(
        self,
        image_size: int,
        image_channel: int,
        patch_size: int,
        self_attn_depth: int,
        cls_attn_depth: int,
        dim: int,
        heads: int,
        alpha: float,
        local_kernel_size: int = 3,
        act_fnc_name: str = "GELU",
        use_bias: bool = False,
        ff_expand_scale: int = 4,
        ff_dropout: float = 0.,
        attention_dropout: float = 0.,
        path_dropout: float = 0.,
        token_dropout: float = 0.,
    ) -> None:
        super().__init__(image_size, image_channel, patch_size)

        # self.conv_proj = 
        
        self.CLS = nn.Parameter(torch.randn(1, 1, dim))
        self.token_dropout = TokenDropout(token_dropout)

        self.self_attn_layers = nn.Sequential(OrderedDict([
            (
                f"self_attn_layer_{idx}",
                XCiTLayer(
                    dim,
                    kernel_size=local_kernel_size,
                    act_fnc_name=act_fnc_name,
                    heads=heads,
                    use_bias=use_bias,
                    ff_expand_scale=ff_expand_scale,
                    ff_dropout=ff_dropout,
                    attention_dropout=attention_dropout,
                    path_dropout=path_dropout,
                    alpha=alpha,
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

        

        # Token dropout
        x = self.token_dropout(x)

        # Expand CLS token ann add position code
        cls_token = repeat(self.CLS, "1 1 d -> b 1 d", b=b)
        x = x + self.position_code
        
        # Self-Attention Layers
        x = self.self_attn_layers(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        
        # Classe Attention Layers
        for cls_layer in self.cls_attn_layers:
            cls_token = cls_layer(cls_token, x)

        return cls_token