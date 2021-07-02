from comvex.utils.base_block import FeedForward
from typing import List, Dict, Optional
from collections import OrderedDict, namedtuple

import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

from comvex.utils import PathDropout, ProjectionHead
from comvex.utils.helpers import name_with_msg, config_pop_argument
from .config import CoaTConfig


CoaTReturnType = namedtuple("CoaTRetureType", "x cls_token")


class ConvolutionalPositionEncoding(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.depth_wise_conv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=dim
        )

    def forward(self, x: torch.Tensor, H: Optional[int] = None, W: Optional[int] = None) -> torch.Tensor:
        r"""
        If `H` and `W` are not given, assume `x` hasn't been flatten and its shape should be (b, c, h, w)
        """
        if H and W:
            x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        x = self.depth_wise_conv(x)

        if H and W:
            x = rearrange(x, "b c h w -> b (h w) c")

        return x


class ConvolutionalRelativePositionEncoding(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: Optional[int],
        head_dim: Optional[int],
        kernel_size_on_heads: Dict[int, int] = { 3: 8 },
    ) -> None:
        super().__init__()

        head_list = list(kernel_size_on_heads.values())
        if heads is None and head_dim is None:
            if any([True if h is None or h <= 0 else False for h in head_list]):
                raise ValueError(
                    "Please specify exact number (integers that are greater than 0) of heads for each kernel size when `heads` and `head_dim` are None."
                )

            self.heads = sum(head_list)
        else:
            self.heads = heads or dim // head_dim
            
            idx_auto = head_list.index(-1)
            if idx_auto != -1:  # If "-1" in the `head_list` exist
                head_list.pop(idx_auto)
                if head_list.index(-1) != -1:  # If there are more than one -1
                    raise ValueError(
                        f"Only one kernel size's number of heads can be specified as -1. Got: {kernel_size_on_heads}"
                    )
                
                # update heads
                kernel_size_on_heads[kernel_size_on_heads.keys()[idx_auto]] = self.heads - sum(head_list)

        self.head_dim = head_dim or dim // self.heads

        assert (
            dim // self.heads == self.head_dim
        ), name_with_msg(f"`dim` ({dim}) can't be divided by `heads` ({self.heads}). Please check `heads`, `head_dim`, or `kernel_size_on_heads`.")

        self.depth_wise_conv_list = nn.ModuleList([
            nn.Conv2d(
                self.head_dim*num_heads,
                self.head_dim*num_heads,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=self.head_dim*num_heads,
            ) for kernel_size, num_heads in kernel_size_on_heads
        ])
        self.split_list = list(kernel_size_on_heads.values())

    def forward(self, x: torch.Tensor, H: Optional[int] = None, W: Optional[int] = None) -> torch.Tensor:
        r"""
        If `H` and `W` are not given, assume `x` hasn't been flatten and its shape should be (b, c, h, w), so does the outputs
        """
        if H and W:
            x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        x_list = torch.split(x, self.split_list, dim=1)
        x_list = [conv(x) for x, conv in zip(x_list, self.depth_wise_conv_list)]
        x = torch.cat(x_list, dim=1)

        if H and W:
            x = rearrange(x, "b c h w -> b (h w) c")

        return x


class FactorizedAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: Optional[int],
        head_dim: Optional[int],
        kernel_size_on_heads: Dict[int, int],
        use_cls: bool = True,
        use_bias: bool = True,
        attention_dropout: float = 0.,
        ff_dropout: float = 0.,
    ) -> None:
        super().__init__()

        assert (
            heads is not None or head_dim is not None
        ), name_with_msg(f"Either `heads` ({heads}) or `head_dim` ({head_dim}) must be specified")

        self.heads = heads if heads is not None else dim // head_dim
        head_dim = head_dim if head_dim is not None else dim // heads

        assert (
            head_dim * self.heads == dim
        ), name_with_msg("Head dimension ({head_dim}) times the number of heads ({self.heads}) must be equal to embedding dimension ({dim})")

        self.relative_position_encoder = ConvolutionalRelativePositionEncoding(
            dim,
            heads,
            head_dim,
            kernel_size_on_heads=kernel_size_on_heads,
        )
        self.QKV = nn.Linear(dim, dim, bias=use_bias)
        self.out = nn.Linear(dim, dim)

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.out_dropout= nn.Dropout(ff_dropout)

        self.scale = head_dim**(-0.5)
        self.use_cls = use_cls

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        h = self.heads

        #
        q, k, v = self.QKV(x).chunk(chunks=3, dim=-1)
        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        k = rearrange(k, "b n (h d) -> b h d n", h=h).softmax(dim=-1)
        v = rearrange(v, "b n (h d) -> b h n d", h=h)

        #
        attention = einsum("b h p n, b h n q -> b h p q", k, v)
        attention = self.attention_dropout(attention)

        #
        if self.use_cls:
            b, n, _, d = v.shape
            relative_position = self.relative_position_encoder(v[:, :, 1:, :], H, W)
            cls_relative_position = torch.zeros(
                (b, n, 1, d),
                dtype=relative_position.dtype,
                device=relative_position.device,
                layout=relative_position.layout
            )
            relative_position = torch.cat([cls_relative_position, relative_position], dim=-2)
        else:
            relative_position = self.relative_position_encoder(v, H, W)

        relative_position = einsum("b h n p, b h n q", q, v)

        #
        out = einsum("b h n p, b h p q -> b h n q", q*self.scale, attention) + relative_position
        out = rearrange("b h n d -> b n (h d)", out)
        out = self.out_linear(out)
        out = self.out_dropout(out)

        return out


class ConvAttentionalModule(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: Optional[int],
        head_dim: Optional[int],
        kernel_size_on_heads: Dict[int, int],
        use_cls: bool = True,
        use_bias: bool = True,
        attention_dropout: float = 0.,
        ff_dropout: float = 0.,
    ) -> None:
        super().__init__()

        assert (
            heads is not None or head_dim is not None
        ), name_with_msg(f"Either `heads` ({heads}) or `head_dim` ({head_dim}) must be specified")

        self.heads = heads if heads is not None else dim // head_dim
        head_dim = head_dim if head_dim is not None else dim // heads

        assert (
            head_dim * self.heads == dim
        ), name_with_msg("Head dimension ({head_dim}) times the number of heads ({self.heads}) must be equal to embedding dimension ({dim})")

        # Add convolutional position encoding in `SerialBlock`, 
        # which differ from `Figure 2` in the paper but aligns the official implementation:
        # https://github.com/mlpc-ucsd/CoaT/blob/main/src/models/coat.py#L211-L214
        # self.general_position_encoder = ConvolutionalPositionEncoding(dim)

        self.relative_position_encoder = ConvolutionalRelativePositionEncoding(
            dim,
            heads,
            head_dim,
            kernel_size_on_heads=kernel_size_on_heads,
        )
        self.QKV = nn.Linear(dim, dim, bias=use_bias)
        self.out = nn.Linear(dim, dim)

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.out_dropout= nn.Dropout(ff_dropout)

        self.scale = head_dim**(-0.5)
        self.use_cls = use_cls
        
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        h = self.heads

        #
        q, k, v = self.QKV(x).chunk(chunks=3, dim=-1)
        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        k = rearrange(k, "b n (h d) -> b h d n", h=h).softmax(dim=-1)
        v = rearrange(v, "b n (h d) -> b h n d", h=h)

        #
        attention = einsum("b h p n, b h n q -> b h p q", k, v)
        attention = self.attention_dropout(attention)

        #
        if self.use_cls:
            b, n, _, d = v.shape
            relative_position = self.relative_position_encoder(v[:, :, 1:, :], H, W)
            cls_relative_position = torch.zeros(
                (b, n, 1, d),
                dtype=relative_position.dtype,
                device=relative_position.device,
                layout=relative_position.layout
            )
            relative_position = torch.cat([cls_relative_position, relative_position], dim=-2)
        else:
            relative_position = self.relative_position_encoder(v, H, W)

        relative_position = einsum("b h n p, b h n q", q, v)

        #
        out = einsum("b h n p, b h p q -> b h n q", q*self.scale, attention) + relative_position
        out = rearrange("b h n d -> b n (h d)", out)
        out = self.out_linear(out)
        out = self.out_dropout(out)

        return out


class SerialBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ff_expand_scale: int = 4,
        path_dropout: float = 0.,
        use_cls: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.conv_position_encoder = ConvolutionalPositionEncoding(dim)

        self.norm_0 = nn.LayerNorm(dim)
        self.conv_attn_module = ConvAttentionalModule(
            dim,
            use_cls=use_cls,
            **kwargs,
        )
        self.path_dropout_0 = PathDropout(path_dropout)

        self.norm_1 = nn.LayerNorm(dim)
        self.ff_block = FeedForward(
            dim,
            ff_expand_scale=ff_expand_scale,
            ff_dropout=kwargs["ff_dropout"],
        )
        self.path_dropout_1 = PathDropout(path_dropout)

        self.use_cls = use_cls

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # Add convolutional position encoding before the ``ConvAttentionalModule`, 
        # which differ from `Figure 2` in the paper but aligns the official implementation:
        # https://github.com/mlpc-ucsd/CoaT/blob/main/src/models/coat.py#L211-L214

        if self.use_cls:
            cls_token, x = x[:, :1, :], x[:, 1:, :]

        x = self.conv_position_encoder(x, H, W)

        if self.use_cls:
            x = torch.cat([cls_token, x], dim=1)

        x = x + self.path_dropout_0(self.conv_attn_module(self.norm_0(x)))

        x = x + self.path_dropout_1(self.ff_block(self.norm_1(x)))

        return x


class ParallelBlock(nn.Module):
    def __init__(self):
        super().__init__()