from comvex.utils.dropout import PathDropout
import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from comvex.utils import MBConvXd, MultiheadAttention, FeedForward
from comvex.utils.helpers import name_with_msg


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
    ):
        super().__init__(in_dim, proj_dim=proj_dim, out_dim=proj_dim, heads=heads, attention_dropout=attention_dropout, ff_dropout=ff_dropout)

        self.pre_height = pre_height
        self.pre_width = pre_width

        self.relative_bias = nn.Parameter(
            torch.randn(heads, pre_height*pre_width),
            requires_grad=True
        )
        self.register_buffer["relative_indices"] = self._get_relative_indices(pre_height, pre_width)
        

    def _get_relative_indices(self, height: int, width: int) -> torch.tensor:
        # torch.nn.functional.grid_sample
        ticks_y = torch.arange(height)
        ticks_x = torch.arange(width)
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

    def forward(self, x: torch.tensor) -> torch.tensor:
        b, c, H, W, h = *x.shape, self.heads
    
        x = rearrange(x, "b c h w -> b (h w) c")  # b n c
        q, k, v = map(lambda proj: proj(x), (self.Q, self.K, self.V))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        relative_indices = repeat(self.relative_indices, "n m -> b h n m", b=b, h=h)
        relative_bias = repeat(self.relative_bias, "h m -> b h n m", b=b, n=n)
        relative_biases = relative_bias.gather(dim=-1, index=relative_indices)

        similarity = einsum("b h n d, b h m d -> b h n m", q, k) + relative_biases  # m=n
        similarity = similarity.softmax(dim=-1)
        similarity = self.attention_dropout(similarity)
        
        out = einsum("b h n m, b h m d -> b h n d", similarity, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.out_dropout(self.out_linear(out))
        out = rearrange(out, "b (h w) c -> b c h w", h=H)

        return self.out_dropout(out)

         
class CoAtNetTransformerBlock(nn.Module):
    def __init__(
        self,
        input_height: int,
        input_width: int,
        in_dim: int,
        out_dim: int,
        expand_scale: int = 4,
        **kwargs
    ):
        super().__init__()

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
        self.path_dropout = PathDropout(kwargs["path_dropout"] if "path_dropout" in kwargs else 0.)
        self.pool = nn.AdaptiveMaxPool2d(input_height//2, input_width//2) if in_dim != out_dim else nn.Identity()
        self.skip = nn.Conv2d(in_dim, out_dim, kernel_size=1) if in_dim != out_dim else nn.Identity()

        self.ff_block = nn.Sequential(
            FeedForward(dim=in_dim, ff_dim_scale=expand_scale, **kwargs),
            PathDropout(kwargs["path_dropout"] if "path_dropout" in kwargs else 0.),
        )

    def forward(self, x):
        # Equation (4) in the official paper with an extra path dropout
        x = self.skip(self.pool(x)) + self.path_dropout(self.attention_block(self.pool(self.norm(x))))
        x = x + self.ff_block(x)

        return x


class CoAtNetConvBlock(nn.Module):
    def __init__(
        self,
        input_height: int,
        input_width: int,
        in_dim: int,
        out_dim: int,
        expand_scale: int = 4,
        **kwargs
    ):
        super().__init__()

        self.norm = nn.BatchNorm2d(in_dim)
        self.mb_conv = MBConvXd(
            in_dim,
            out_dim,
            expand_scale=expand_scale,
            first_pixel_wise_conv_stride=2 if in_dim != out_dim else 1,
            **kwargs
        )

        self.skip = nn.Sequential(
            nn.AdaptiveMaxPool2d(input_height//2, input_width//2),
            nn.Conv2d(in_dim, out_dim, kernel_size=1)
        ) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        # Equation (5) in the offical paper
        x = self.skip(x) + self.mb_conv(self.norm(x))

        return x