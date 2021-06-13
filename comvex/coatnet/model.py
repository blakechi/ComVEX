from typing import List, OrderedDict, Union

import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

from comvex.utils import MBConvXd, MultiheadAttention, PathDropout, ProjectionHead
from comvex.utils.helpers import name_with_msg, config_pop_argument
from .config import CoAtNetConfig


class CoAtNetRelativeAttention(MultiheadAttention):
    def __init__(
        self,
        pre_height: int,
        pre_width: int,
        in_dim: int,
        proj_dim: int,
        heads: int,
        attention_dropout: float = 0.,
        ff_dropout: float = 0.,
    ) -> None:
        super().__init__(in_dim, proj_dim=proj_dim, out_dim=proj_dim, heads=heads, attention_dropout=attention_dropout, ff_dropout=ff_dropout)

        self.pre_height = pre_height
        self.pre_width = pre_width

        self.relative_bias = nn.Parameter(
            torch.randn(heads, int((2*pre_height - 1)*(2*pre_width - 1))),
            requires_grad=True
        )
        self.register_buffer("relative_indices", self._get_relative_indices(pre_height, pre_width))

    def forward(self, x: torch.tensor) -> torch.tensor:
        b, c, H, W, h = *x.shape, self.heads
    
        #
        x = rearrange(x, "b c h w -> b (h w) c")  # b n c
        q, k, v = map(lambda proj: proj(x), (self.Q, self.K, self.V))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        # relative biases
        if H == self.pre_height and W == self.pre_width:
            relative_indices = self.relative_indices
            relative_bias = self.relative_bias
        else:
            relative_indices = self._get_relative_indices(H, W)
            relative_bias = self._interpolate_relative_bias(H, W)

        relative_indices = repeat(relative_indices, "n m -> b h n m", b=b, h=h)
        relative_bias = repeat(relative_bias, "h r -> b h n r", b=b, n=H*W)  # r: number of relative biases, (2*H - 1)*(2*W - 1)
        relative_biases = relative_bias.gather(dim=-1, index=relative_indices)

        # similarity
        similarity = einsum("b h n d, b h m d -> b h n m", q, k) + relative_biases  # m=n
        similarity = similarity.softmax(dim=-1)
        similarity = self.attention_dropout(similarity)
        
        # 
        out = einsum("b h n m, b h m d -> b h n d", similarity, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.out_dropout(self.out_linear(out))
        out = rearrange(out, "b (h w) c -> b c h w", h=H)

        return out

    def _get_relative_indices(self, height: int, width: int) -> torch.tensor:
        height, width = int(height), int(width)
        ticks_y, ticks_x = torch.arange(height), torch.arange(width)
        grid_y, grid_x = torch.meshgrid(ticks_y, ticks_x)
        out = torch.empty(height*width, height*width).fill_(float("nan"))

        for idx_y in range(height):
            for idx_x in range(width):
                rel_indices_y = grid_y - idx_y + height
                rel_indices_x = grid_x - idx_x + width
                flatten_indices = (rel_indices_y*width + rel_indices_x).flatten()
                out[idx_y*width + idx_x] = flatten_indices

        assert (
            not out.isnan().any()
        ), name_with_msg(self, "`relative_indices` have blank indices")
        
        assert (
            (out >= 0).all()
        ), name_with_msg(self, "`relative_indices` have negative indices")

        return out.to(torch.long)

    def _interpolate_relative_bias(self, height: int, width: int) -> torch.Tensor:
        out = rearrange(self.relative_bias, "h (n m) -> 1 h n m", n=(2*self.pre_height - 1))
        out = nn.functional.interpolate(out, size=(2*height - 1, 2*width - 1), mode="bilinear", align_corners=True)

        return rearrange(out, "1 h n m -> h (n m)")

    def _update_relative_bias_and_indices(self, height: int, width: int) -> None:
        r"""
        For possible input's height or width changes in inference.
        """

        self.relative_indices = self._get_relative_indices(height, width)
        self.relative_bias = self._interpolate_relative_bias(height, width)
        

class CoAtNetTransformerBlock(nn.Module):
    def __init__(
        self,
        input_height: int,
        input_width: int,
        in_dim: int,
        out_dim: int,
        expand_scale: int = 4,
        use_downsampling: bool = False,
        **kwargs
    ):
        super().__init__()
        path_dropout = kwargs.pop("path_dropout")

        self.norm = nn.Sequential(
            Rearrange("b c h w -> b h w c"),
            nn.LayerNorm(in_dim),
            Rearrange("b h w c -> b c h w"),
        )
        self.attention_block = CoAtNetRelativeAttention(
            input_height,
            input_width,
            in_dim,
            out_dim,
            **kwargs
        )
        self.attention_path_dropout = PathDropout(path_dropout)
        self.pool = nn.MaxPool2d((2, 2)) if use_downsampling else nn.Identity()
        self.skip = nn.Conv2d(in_dim, out_dim, kernel_size=1) if use_downsampling else nn.Identity()

        self.ff_block = nn.Sequential(
            nn.Conv2d(out_dim, out_dim*expand_scale, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_dim*expand_scale, out_dim, kernel_size=1),
        )
        self.ff_path_dropout = PathDropout(path_dropout)

    def forward(self, x):
        # Equation (4) in the official paper with an extra path dropout
        x = self.skip(self.pool(x)) + self.attention_path_dropout(self.attention_block(self.pool(self.norm(x))))
        x = x + self.ff_path_dropout(self.ff_block(x))

        return x


class CoAtNetConvBlock(nn.Module):
    def __init__(
        self,
        input_height: int,
        input_width: int,
        in_dim: int,
        out_dim: int,
        expand_scale: int = 4,
        use_downsampling: bool = False,
        **kwargs
    ):
        super().__init__()

        self.norm = nn.BatchNorm2d(in_dim)
        self.mb_conv = MBConvXd(
            in_dim,
            out_dim,
            expand_scale=expand_scale,
            first_pixel_wise_conv_stride=2 if use_downsampling else 1,
            **kwargs
        )
        self.path_dropout = PathDropout(kwargs["path_dropout"] if "path_dropout" in kwargs else 0.)

        self.skip = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(in_dim, out_dim, kernel_size=1)
        ) if use_downsampling else nn.Identity()

    def forward(self, x):
        # Equation (5) in the offical paper with an extra path dropout
        x = self.skip(x) + self.path_dropout(self.mb_conv(self.norm(x)))

        return x


class CoAtNetBackbone(nn.Module):
    def __init__(
        self,
        image_height: int,
        image_width: int,
        image_channel: int,
        num_blocks_in_layers: List[int],
        block_type_in_layers: List[int],
        num_channels_in_layers: Union[List[int], int],
        expand_scale_in_layers: Union[List[int], int],
        heads: int = 32,
        ff_dropout: float = 0.,
        attention_dropout: float = 0.,
        path_dropout: float = 0.,
    ) -> None:
        super().__init__()

        assert (
            len(num_blocks_in_layers) == 5
        ), name_with_msg(self, "The length of `num_blocks_in_layers` must be 5")

        if isinstance(num_channels_in_layers, list):
            assert (
                len(num_channels_in_layers) == 5
            ), name_with_msg(self, "The length of `num_channels_in_layers` must be 5")
        else:
            begin_channel = int(num_channels_in_layers)
            num_channels_in_layers = [int(begin_channel // (2**layer_idx)) for layer_idx in range(0, 5)]

        # We ignore `S0` here, so the length of the below lists should be 4
        assert (
            len(block_type_in_layers) == 4
        ), name_with_msg(self, "The length of `block_type_in_layers` must be 4")

        if isinstance(expand_scale_in_layers, list):
            assert (
                len(expand_scale_in_layers) == 4
            ), name_with_msg(self, "The length of `expand_scale_in_layers` must be 4")
        else:
            expand_scale = int(expand_scale_in_layers)
            expand_scale_in_layers = [expand_scale for _ in range(4)]

        height_in_layers = [int(image_height / (2**layer_idx)) for layer_idx in range(1, 6)]
        width_in_layers = [int(image_width / (2**layer_idx)) for layer_idx in range(1, 6)]

        kwargs = {}
        kwargs["heads"] = heads
        kwargs["ff_dropout"] = ff_dropout
        kwargs["attention_dropout"] = attention_dropout
        kwargs["path_dropout"] = path_dropout

        # Layers
        self.s_0 = nn.Sequential(*[
            nn.Conv2d(
                image_channel,
                num_channels_in_layers[0],
                kernel_size=3,
                stride=2,
                padding=1
            ) if idx == 0 else nn.Conv2d(
                num_channels_in_layers[0],
                num_channels_in_layers[0],
                kernel_size=3,
                padding=1
            ) for idx in range(num_blocks_in_layers[0])
        ])
        self.s_1 = self._build_layer("s_1",                                                   # layer name
            height_in_layers[1], width_in_layers[1],                                          # input size
            num_channels_in_layers[0], num_channels_in_layers[1], expand_scale_in_layers[0],  # dimension-related
            num_blocks_in_layers[1], block_type_in_layers[0],                                 # block-related
            **kwargs
        )
        self.s_2 = self._build_layer("s_2",                                                   # layer name
            height_in_layers[2], width_in_layers[2],                                          # input size
            num_channels_in_layers[1], num_channels_in_layers[2], expand_scale_in_layers[1],  # dimension-related
            num_blocks_in_layers[2], block_type_in_layers[1],                                 # block-related
            **kwargs
        )
        self.s_3 = self._build_layer("s_3",                                                   # layer name
            height_in_layers[3], width_in_layers[3],                                          # input size
            num_channels_in_layers[2], num_channels_in_layers[3], expand_scale_in_layers[2],  # dimension-related
            num_blocks_in_layers[3], block_type_in_layers[2],                                 # block-related
            **kwargs
        )
        self.s_4 = self._build_layer("s_4",                                                   # layer name
            height_in_layers[4], width_in_layers[4],                                          # input size
            num_channels_in_layers[3], num_channels_in_layers[4], expand_scale_in_layers[3],  # dimension-related
            num_blocks_in_layers[4], block_type_in_layers[3],                                 # block-related
            **kwargs
        )

        self.pooler = Reduce("b c h w -> b c", "mean")  # `global_pool` in the official paper

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.s_0(x)
        x = self.s_1(x)
        x = self.s_2(x)
        x = self.s_3(x)
        x = self.s_4(x)

        return self.pooler(x)  

    def _build_layer(
        self,
        layer_name: str,
        height: int,
        width: int,
        in_channel: int,
        out_channel: int,
        expand_scale: int,
        num_blocks: int,
        block_type: str,
        **kwargs
    ) -> nn.Module:
        if block_type == "C":
            core_block = CoAtNetConvBlock
        elif block_type == "T":
            core_block = CoAtNetTransformerBlock
        else:
            raise ValueError(f"Block type: '{block_type}' doesn't exist. Please choose between 'C' or 'T'")

        return nn.Sequential(OrderedDict([
            (f"{layer_name}_{idx}", core_block(
                input_height=height,
                input_width=width,
                in_dim=in_channel if idx == 0 else out_channel,
                out_dim=out_channel,
                expand_scale=expand_scale,
                use_downsampling=True if idx == 0 else False,
                **kwargs
            )) for idx in range(num_blocks)
        ]))


class CoAtNetWithLinearClassifier(CoAtNetBackbone):
    def __init__(self, config: CoAtNetConfig = None):
        num_classes = config_pop_argument(config, "num_classes")
        pred_act_fnc_name = config_pop_argument(config, "pred_act_fnc_name")
        super().__init__(**config.__dict__)

        self.proj_head = ProjectionHead(
            config.num_channels_in_layers[-1],
            num_classes,
            pred_act_fnc_name,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)

        return self.proj_head(x)