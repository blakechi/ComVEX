import enum
from os import name
from typing import List, Dict, Optional, Tuple

import torch
from torch import nn, einsum
from einops import rearrange, repeat

from comvex.vit import ViTBase
from comvex.utils import FeedForward, PathDropout, ProjectionHead, PatchEmbeddingXd
from comvex.utils.helpers import name_with_msg, config_pop_argument
from .config import CoaTLiteConfig


class CoaTBase(ViTBase):
    def __init__(
        self,
        image_size: int,
        image_channel: int,
        patch_size: int,
        num_layers_in_stages: List[int],
        num_channels: List[int],
        expand_scales: List[int],
        kernel_size_on_heads: Dict[int, int],
        heads: Optional[int] = None,
    ) -> None:
        super().__init__(image_size, image_channel, patch_size, use_patch_and_flat=False)

        self.image_channel = image_channel
        self.num_stages = len(num_layers_in_stages)
        self.num_layers_in_stages = num_layers_in_stages
        self.num_channels = num_channels
        self.expand_scales = expand_scales
        self.patch_sizes = [self.patch_size, *((2, )*(self.num_stages - 1))]

        if heads is not None:
            assert (
                heads == sum(kernel_size_on_heads.values())
            ), name_with_msg(f"Number of heads should be equal for `heads` ({heads}) and the sum of values of `kernel_size_on_heads` ({sum(kernel_size_on_heads.values())})")
        self.heads = heads or sum(kernel_size_on_heads.values())
        self.kernel_size_on_heads = kernel_size_on_heads

class ConvolutionalPositionEncoding(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        use_cls: bool = True,
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
        
        self.use_cls = use_cls

    def forward(self, x: torch.Tensor, H: Optional[int] = None, W: Optional[int] = None) -> torch.Tensor:
        r"""
        If `H` and `W` are not given, assume `x` hasn't been flatten and its shape should be (b, c, h, w)
        """
        if self.use_cls:
            cls_token, x = x[:, :1, :], x[:, 1:, :]

        if H and W:
            x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        x = self.depth_wise_conv(x)

        if H and W:
            x = rearrange(x, "b c h w -> b (h w) c")
        
        if self.use_cls:
            x = torch.cat([cls_token, x], dim=1)

        return x


class ConvolutionalRelativePositionEncoding(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: Optional[int],
        head_dim: Optional[int],
        kernel_size_on_heads: Dict[int, int] = { 3: 2, 5: 3, 7: 3 },  # From: https://github.com/mlpc-ucsd/CoaT/blob/main/src/models/coat.py#L358
        use_cls: bool = True,
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
            ) for kernel_size, num_heads in kernel_size_on_heads.items()
        ])
        self.split_list = [num_heads*self.head_dim for num_heads in kernel_size_on_heads.values()]

        self.use_cls = use_cls

    def forward(self, q: torch.Tensor, v: torch.Tensor, H: Optional[int] = None, W: Optional[int] = None) -> torch.Tensor:
        r"""
        If `H` and `W` are not given, assume `x` hasn't been flatten and its shape should be (b, c, h, w), so does the outputs
        """

        if H and W:
            b, p, n, d = v.shape  # p for heads

        if self.use_cls:
            v = v[:, :, 1:, :]

        if H and W:
            v = rearrange(v, "b p (h w) d -> b (p d) h w", h=H, w=W)

        v_list = torch.split(v, self.split_list, dim=1)
        v_list = [conv(v) for v, conv in zip(v_list, self.depth_wise_conv_list)]
        v = torch.cat(v_list, dim=1)

        if H and W:
            v = rearrange(v, "b (p d) h w -> b p (h w) d", p=p, d=d)

        if self.use_cls:
            cls_relative_position = torch.zeros(
                (b, p, 1, d),
                dtype=v.dtype,
                device=v.device,
                layout=v.layout
            )
            v = torch.cat([cls_relative_position, v], dim=-2)

        return q*v


class FactorizedAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size_on_heads: Dict[int, int],
        heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        use_cls: bool = True,
        use_bias: bool = True,
        conv_relative_postion_encoder: Optional[nn.Module] = None,
        attention_dropout: float = 0.,
        ff_dropout: float = 0.,
    ) -> None:
        super().__init__()

        assert (
            heads is not None or head_dim is not None
        ), name_with_msg(self, f"Either `heads` ({heads}) or `head_dim` ({head_dim}) must be specified")

        self.heads = heads if heads is not None else dim // head_dim
        head_dim = head_dim if head_dim is not None else dim // heads

        assert (
            head_dim * self.heads == dim
        ), name_with_msg(self, f"Head dimension ({head_dim}) times the number of heads ({self.heads}) must be equal to embedding dimension ({dim})")

        self.relative_position_encoder = ConvolutionalRelativePositionEncoding(
            dim,
            heads,
            head_dim,
            kernel_size_on_heads=kernel_size_on_heads,
            use_cls=use_cls
        ) if conv_relative_postion_encoder is None else conv_relative_postion_encoder

        self.QKV = nn.Linear(dim, 3*dim, bias=use_bias)
        self.out_linear = nn.Linear(dim, dim)

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
        relative_position = self.relative_position_encoder(q, v, H, W)

        #
        out = einsum("b h n p, b h p q -> b h n q", q*self.scale, attention) + relative_position
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.out_linear(out)
        out = self.out_dropout(out)

        return out


class ConvAttentionalModule(nn.Module):
    def __init__(
        self,
        dim: int,
        use_cls: bool = True,
        use_conv_position_encoder: bool = False,
        conv_position_encoder: Optional[nn.Module] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # Set `conv_position_encoder` to optional since the official implementation adds convolutional position encoding in `SerialBlock`,
        # which differ from `Figure 2` in the paper
        # https://github.com/mlpc-ucsd/CoaT/blob/main/src/models/coat.py#L211-L214
        use_conv_position_encoder = True if conv_position_encoder is not None else use_conv_position_encoder
        if use_conv_position_encoder:
            self.conv_position_encoder = ConvolutionalPositionEncoding(
                dim,
                use_cls=use_cls
            ) if conv_position_encoder is None else conv_position_encoder

        self.factorized_attn = FactorizedAttention(dim=dim, use_cls=use_cls, **kwargs)
        
        self.use_conv_position_encoder = use_conv_position_encoder
        self.use_cls = use_cls

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        if self.use_conv_position_encoder:
            x = self.conv_position_encoder(x, H, W)

        x = self.factorized_attn(x, H, W)

        return x


class CoaTSerialBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ff_expand_scale: int = 4,
        path_dropout: float = 0.,
        conv_position_encoder: Optional[nn.Module] = None,
        use_cls: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.conv_position_encoder = ConvolutionalPositionEncoding(
            dim,
            use_cls=use_cls
        ) if conv_position_encoder is None else conv_position_encoder

        self.norm_0 = nn.LayerNorm(dim)
        self.conv_attn_module = ConvAttentionalModule(
            dim,
            use_cls=use_cls,
            use_conv_position_encoder=False,
            conv_position_encoder=None,
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

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # Add convolutional position encoding before the ``ConvAttentionalModule`, 
        # which differ from `Figure 2` in the paper but aligns the official implementation:
        # https://github.com/mlpc-ucsd/CoaT/blob/main/src/models/coat.py#L211-L214
        x = self.conv_position_encoder(x, H, W)

        x = x + self.path_dropout_0(self.conv_attn_module(self.norm_0(x), H, W))
        x = x + self.path_dropout_1(self.ff_block(self.norm_1(x)))

        return x


class CoaTParallelBlock(nn.Module):
    def __init__(
        self,
        num_feature_maps: int,
        dim: int,
        ff_expand_scale: int = 4,
        path_dropout: float = 0.,
        conv_position_encoder: Optional[nn.Module] = None,
        use_cls: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.conv_position_encoder = nn.ModuleList([
            ConvolutionalPositionEncoding(
                dim,
                use_cls=use_cls
            ) if conv_position_encoder is None else conv_position_encoder for _ in range(num_feature_maps)
        ])

        self.norm_0 = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(num_feature_maps)
        ])
        self.conv_attn_module = nn.ModuleList([
            ConvAttentionalModule(
                dim,
                use_cls=use_cls,
                use_conv_position_encoder=False,
                conv_position_encoder=None,
                **kwargs,
            ) for _ in range(num_feature_maps)
        ])
        self.path_dropout_0 = nn.ModuleList([
            PathDropout(path_dropout)
        ])

        self.norm_1 = nn.ModuleList([
            nn.LayerNorm(dim)
        ])
        self.ff_block = nn.ModuleList([
            FeedForward(
                dim,
                ff_expand_scale=ff_expand_scale,
                ff_dropout=kwargs["ff_dropout"],
            ) for _ in range(num_feature_maps)
        ])
        self.path_dropout_1 = nn.ModuleList([
            PathDropout(path_dropout)
        ])

    def forward(self, *args: List[torch.Tensor], sizes: Tuple[Tuple[int, int]]) -> List[torch.Tensor]:
        num_inputs = len(args)
        assert (
            num_inputs == len(self.serial_block_list)
        ), name_with_msg(self, f"The number of inputs ({num_inputs}) should be aligned with the number of feature maps ({len(self.serial_block_list)})")

        #
        args = [conv_position_encoder(x, H, W) for x, H, W, conv_position_encoder in zip(args, sizes, self.conv_position_encoder)]

        #
        args = [norm(x) for x, norm in zip(args, self.norm_0)]
        args = [conv_attn_module(x, H, W) for x, H, W, conv_attn_module in zip(args, sizes, self.conv_attn_module)]

        for idx in range(num_inputs):
            args[idx] = torch.stack([self.interpolate(x, size=sizes[idx]) for x in args], dim=0).sum(dim=0)

        args = [x + path_dropout(x) for x, path_dropout in zip(args, self.path_dropout_0)]

        #
        args = [norm(x) for x, norm in zip(args, self.norm_1)]
        args = [ff_block(x) for x, ff_block in zip(args, self.ff_block)]
        args = [x + path_dropout(x) for x, path_dropout in zip(args, self.path_dropout_1)]

        return args

    def interpolate(self, x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        return nn.functional.interpolate(x, size=size, mode="bilinear")


class CoaTLiteBackbone(CoaTBase):
    def __init__(
        self,
        image_channel: int, 
        image_size: int, 
        patch_size: int,
        num_layers_in_stages: List[int],
        num_channels: List[int],
        expand_scales: List[int],
        heads: Optional[int] = None,
        kernel_size_on_heads: Dict[int, int] = { 3: 2, 5: 3, 7: 3 },  # From: https://github.com/mlpc-ucsd/CoaT/blob/main/src/models/coat.py#L358
        use_bias: bool = True,
        attention_dropout: float = 0.,
        ff_dropout: float = 0.,
        path_dropout: float = 0.,
    ) -> None:
        super().__init__(
            image_channel=image_channel,
            image_size=image_size,
            patch_size=patch_size,
            num_layers_in_stages=num_layers_in_stages,
            num_channels=num_channels,
            expand_scales=expand_scales,
            heads=heads,
            kernel_size_on_heads=kernel_size_on_heads,
        )
        
        kwargs = {}
        kwargs["use_bias"] = use_bias
        kwargs["attention_dropout"] = attention_dropout
        kwargs["ff_dropout"] = ff_dropout
        kwargs["path_dropout"] = path_dropout

        self.stages = nn.ModuleList([
            self._build_stage(idx, **kwargs) for idx in range(len(num_layers_in_stages))
        ])
        for stage_idx, channels in enumerate(self.num_channels):
            self.register_parameter(f"cls_token_{stage_idx}", nn.Parameter(torch.randn(1, 1, channels), requires_grad=True))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        b = x.shape[0]
        feature_maps = {}

        for idx in range(self.num_stages):
            cls_token = self.get_parameter(f"cls_token_{idx}")
            patch_embedding, serial_blocks = self.stages[idx]

            cls_token = repeat(cls_token, "1 1 d -> b 1 d", b=b)
            x, (H, W) = patch_embedding(x)
            x = torch.cat([cls_token, x], dim=1)

            for block in serial_blocks:
                x = block(x, H, W)

            cls_token, x = x[:, :1, :], x[:, 1:, :]
            x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
            feature_maps[f"cls_token_{idx}"] = cls_token
            feature_maps[f"feature_map_{idx}"] = x

        return feature_maps

    @torch.jit.ignore
    def _build_stage(self, stage_idx: int, **kwargs):
        return nn.ModuleList([
            PatchEmbeddingXd(
                image_channel=self.num_channels[stage_idx - 1] if stage_idx > 0 else self.image_channel,
                embedding_dim=self.num_channels[stage_idx],
                patch_size=self.patch_sizes[stage_idx],
            ),
            nn.ModuleList([
                CoaTSerialBlock(
                    dim=self.num_channels[stage_idx],
                    heads=self.heads,
                    kernel_size_on_heads=self.kernel_size_on_heads,
                    ff_expand_scale=self.expand_scales[stage_idx],
                    **kwargs,
                ) for _ in range(self.num_layers_in_stages[stage_idx])
            ])
        ])

    @torch.jit.ignore
    def no_weight_decay(self) -> List[str]:
        return [f"cls_token_{idx}" for idx in range(self.num_stages)] + ["bias", "LayerNorm.weight"]


class CoaTLiteWithLinearClassifier(CoaTLiteBackbone):
    def __init__(self, config: Optional[CoaTLiteConfig] = None) -> None:
        num_classes = config_pop_argument(config, "num_classes")
        pred_act_fnc_name = config_pop_argument(config, "pred_act_fnc_name")
        super().__init__(**config.__dict__)

        self.proj_head = ProjectionHead(
            self.num_channels[-1],
            num_classes,
            pred_act_fnc_name,
        )

    def forward(self, x):
        feature_maps = super().forward(x)
        cls_token = feature_maps[f"cls_token_{self.num_stages - 1}"]

        return self.proj_head(cls_token)