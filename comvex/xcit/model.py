from typing import Optional, Tuple
from collections import OrderedDict

import torch
from torch import nn, einsum
from einops import rearrange, repeat
from torch.nn.modules.batchnorm import BatchNorm2d

from comvex.vit import ViTBase
from comvex.utils import MLP, TokenDropout, ProjectionHead
from comvex.utils.helpers.functions import config_pop_argument
from .config import XCiConfig


class CrossCovarianceAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        attention_dropout: float = 0.,
        ff_dropout: float = 0.,
        use_bias: bool = False,
        dtype = torch.float32,
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
        pre_activation: bool = False,
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
        self.norm = BatchNorm2d(out_channel)
        self.pre_activation = pre_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_activation:  # If-Else will be removed by scripting or tracing
            x = self.act_fnc(x)
            x = self.norm(x)
            x = self.depth_wise_conv_0(x)
        else:
            x = self.depth_wise_conv_0(x)
            x = self.act_fnc(x)
            x = self.norm(x)

        x = self.depth_wise_conv_1(x)

        return x


class XCiTLayer(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()