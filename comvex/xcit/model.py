from functools import partial
from typing import Optional
from math import log2

import torch
from torch import nn, einsum
from einops import rearrange, repeat


from comvex.vit import ViTBase
from comvex.cait import ClassAttentionLayer
from comvex.utils import FeedForward, TokenDropout, ProjectionHead, LayerScale, PositionEncodingFourier
from comvex.utils.helpers import config_pop_argument, get_act_fnc, name_with_msg
from .config import XCiTConfig


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
        # b, n, e = x.shape

        q, k, v = self.QKV(x).chunk(chunks=3, dim=-1)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h d n", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

        q = torch.nn.functional.normalize(q, dim=-2)
        k = torch.nn.functional.normalize(k, dim=-1)*self.temperature
        attention = einsum("b h p n, b h n q -> b h p q", k, q)  # p == q == d
        attention = attention.softmax(dim=-1)
        attention = self.attention_dropout(attention)
        
        out = einsum("b h n p, b h p q -> b h n q", v, attention)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.out_dropout(self.out_linear(out))

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

        self.act_fnc = get_act_fnc(act_fnc_name)()
        self.norm = nn.BatchNorm2d(out_channel, eps=epsilon, momentum=momentum, affine=affine)

    def forward(self, x: torch.Tensor, pH: int, pW: int) -> torch.Tensor:
        x = rearrange(x, "b (h w) c -> b c h w", h=pH, w=pW)
        x = self.depth_wise_conv_0(x)
        x = self.act_fnc(x)
        x = self.norm(x)
        x = self.depth_wise_conv_1(x)
        x = rearrange(x, "b c h w -> b (h w) c")

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
        )

    def forward(self, x: torch.Tensor, pH: int, pW: int) -> torch.Tensor:
        self.lpi(x, pH, pW)
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
        upsampling_mode: Optional[str] = None
    ) -> None:
        super().__init__(image_size, image_channel, patch_size, use_patch_and_flat=False)

        self.patch_proj = torch.jit.script(PatchProjection(image_channel, self.patch_size, dim))
        self.position_code = torch.jit.script(PositionEncodingFourier(dim=dim, to_flatten=True))
        self.CLS = nn.Parameter(torch.randn(1, 1, dim))
        self.token_dropout = TokenDropout(token_dropout)

        self.self_attn_layers = nn.ModuleList([
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
            ) for _ in range(self_attn_depth)
        ])

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

        self.upsampling = nn.Upsample(size=image_size, mode=upsampling_mode) if upsampling_mode is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]  # b, c, h, w = x.shape 

        if self.upsampling:
            x = self.upsampling(x)

        # Patch projection and flatten
        x = self.patch_proj(x)  # (b, c, H, W) -> (b, d, num_patches, num_patches)
        pH, pW = x.shape[-2], x.shape[-1]  # Number of patches as height and width
        x = rearrange(x, "b c h w -> b (h w) c")
        
        # Token dropout
        x = self.token_dropout(x)

        # Add position code
        x = x + self.position_code(b, pH, pW)
        
        # Self-Attention Layers (CrossCovarianceAttention)
        for self_attn_layer in self.self_attn_layers:
            x = self_attn_layer(x, pH, pW)
        
        # Classe Attention Layers
        cls_token = repeat(self.CLS, "1 1 d -> b 1 d", b=b)
        for cls_layer in self.cls_attn_layers:
            cls_token = cls_layer(cls_token, x)

        return cls_token


class XCiTWithLinearClassifier(XCiTBackbone):
    def __init__(self, config: Optional[XCiTConfig] = None) -> None:
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