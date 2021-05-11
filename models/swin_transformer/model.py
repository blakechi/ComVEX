from typing import Optional, Tuple
import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from models.vit import ViTBase
from models.utils import Residual, LayerNorm, FeedForward


class WindowAttentionBase(nn.Module):
    def __init__(self, *, dim, heads, window_size, head_dim=None, dtype=torch.float32):
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.heads = heads
        self.head_dim = head_dim if head_dim is not None else dim // heads

        assert (
            self.head_dim * self.heads == self.dim
        ), "Head dimension times the number of heads must be equal to embedding dimension"

        self.relative_position = nn.Parameter(
            torch.empty([self.heads, (2*self.window_size[0] - 1)*(2*self.window_size[1] - 1)], requires_grad=True)
        )
        self.register_buffer("relative_position_index", self._get_relative_position_index())

        self.scale = self.head_dim ** (-0.5)
        self.mask_value = -torch.finfo(dtype).max  # pytorch default float type

    def split_into_windows(self, x):
        x = rearrange(x, "b (p h) (q w) c -> b h w p q c", p=self.window_size, q=self.window_size)
        x = rearrange(x, "b h w (p q) c -> b h w n c")

        return x

    def merge_windows(self, x):
        x = rearrange(x, "b h w (p q) c -> b h w p q c", p=self.window_size)
        x = rearrange(x, "b h w p q c -> b (p h) (q w) c")

        return x

    def get_relative_position_index(self) -> torch.Tensor:
        r"""
        Reference from: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py

        Example: When `window_size` == (2, 2)
        >> out = tensor([[4, 3, 1, 0],
                         [5, 4, 2, 1],
                         [7, 6, 4, 3],
                         [8, 7, 5, 4]])
        >> out = out.view(-1)
        """

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        return rearrange(relative_position_index, "... -> (...)")

    def _init_weights(self, m):
        if isinstance(m, nn.Parameter):
            # Reference from: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L110
            nn.init.trunc_normal_(m, std=0.02)


class WindowAttention(WindowAttentionBase):
    def __init__(self, *, dim, heads, window_size, attention_dropout=0.0, head_dim=None, dtype=torch.float32, **kwargs):
        super().__init__(dim, heads, window_size, head_dim=head_dim, dtype=dtype)

        self.qkv = nn.Linear(dim, 3*dim)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.out_linear = nn.Linear(dim, dim)

        self.apply(self._init_weights)

    def forward(self, x, attention_mask=None):
        return self._forward(x, attention_mask)

    def _forward(self, x, attention_mask=None):
        b, H, W, c, g, p = *x.shape, self.heads, self.window_size

        x = self.split_into_windows(x)

        q, k, v = self.qkv(x).chunk(chunks=3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b h w n (g d) -> b g h w n d", g=g), (q, k, v))

        q = q*self.scale
        similarity = einsum("b g h w n d, b g h w m d -> b g h w n m", q, k)

        relative_position_bias = self.relative_position[:, self.relative_position_index]
        relative_position_bias = rearrange(relative_position_bias, "g (n m) -> 1 g 1 1 n m")
        similarity = similarity + relative_position_bias

        if attention_mask is not None: 
            similarity.masked_fill_(attention_mask, self.mask_value)

        similarity = similarity.softmax(dim=-1) 
        similarity = self.attention_dropout(similarity)

        out = einsum("b g h w n m, b g h w m d -> b g h w n d", similarity, v)
        out = rearrange(out, "b g h w n d -> b h w n (g d)")
        out = self.merge_windows(out)

        return self.out_linear(out)


class ShiftWindowAttention(WindowAttention):
    def __init__(self, *, dim, heads, window_size, shifts, input_resolution, **kwargs):
        super().__init__(dim, heads, window_size, **kwargs)

        self.shifts = shifts
        self.register_buffer("shifted_attention_mask", self._get_shifted_attnetion_mask(input_resolution))

    def forward(self, x, attention_mask=None):
        # b, H, W, c = x.shape

        x = torch.roll(x, (-self.shifts, -self.shifts), (-3, -2))

        attention_mask = attention_mask | self.shifted_attnetion_mask if attention_mask is not None else self.shifted_attention_mask
        x = self._forward(x, attention_mask)

        x = torch.roll(x, (self.shifts, self.shifts), (-3, -2))

        return x

    def _get_shifted_attnetion_mask(self, input_resolution):
        # Reference from: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L210

        H, W = input_resolution
        image_mask = torch.zeros([1, H, W, 1])
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shifts),
                    slice(-self.shifts, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shifts),
                    slice(-self.shifts, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                image_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = self.split_into_windows(image_mask)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        apttn_mask = attn_mask != 0

        return attn_mask


class SwinTransformerBlock(nn.Module):
    def __init__(
        self, 
        dim, 
        window_size, 
        heads, 
        shifts=None, 
        ff_dim=None, 
        pre_norm=False, 
        **kwargs
    ):

        self.attention_block = LayerNorm(
            Residual(
                ShiftWindowAttention(
                    dim=dim, heads=heads, window_size=window_size, shifts=shifts, **kwargs
                ) if shifts is not None else WindowAttention(
                    dim=dim, heads=heads, window_size=window_size, **kwargs
                )
            ),
            dim=dim,
            use_pre_norm=pre_norm
        )
        self.ff_block = LayerNorm(
            Residual(
                FeedForward(
                    dim=dim, hidden_dim=ff_dim if ff_dim is not None else 4*dim, **kwargs
                )
            ),
            dim=dim,
            use_pre_norm=pre_norm
        )

    def forward(self, x, attention_mask):
        x = self.attention_block(x, attention_mask)

        return self.ff_block(x)


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.merge = Rearrange("b (h p) (w q) c -> b h w (p q)", p=2, q=2)
        self.proj_head = nn.Sequential(
            nn.LayerNorm(4*dim),
            nn.Linear(4*dim, 2*dim, bias=False)
        )

    def forward(self, x):
        x = self.merge(x)

        return self.proj_head(x)

    
