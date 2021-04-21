import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from models.utils import Residual, LayerNorm


class BoTMHSA(nn.Module):
    def __init__(self, *, input_shape, heads, head_dim=None, **kwargs):
        super().__init__()

        dim, width, height = input_shape
        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim if head_dim is not None else dim // heads

        assert (
            self.head_dim * self.heads == self.dim
        ), "Head dimension times the number of heads must be equal to embedding dimension"

        self.QKV = nn.Conv2d(dim, 3*dim, 1)
        self.height_relative_pos = nn.Parameter(torch.empty(height, 1, dim))
        self.width_relative_pos = nn.Parameter(torch.empty(width, dim, 1))
        nn.init.kaiming_uniform_(self.height_relative_pos)
        nn.init.kaiming_uniform_(self.width_relative_pos)

        self.scale = self.head_dim ** (-0.5)

    def forward(x):
        b, e, h, w, p, device = *x.shape, self.heads, x.device

        q, k, v = self.QKV(x).chunk(chunks=3, dim=-3)
        q, k, v = map(lambda t: rearrange(t, "b (p d) h w -> b p (h w) d", p=p), (q, k, v))

        relative_pos = rearrange(self.height_relative_pos + self.width_relative_pos, "h w (p d) -> 1 p (h w) d", p=p)

        # Note: scaling doesn't be mention in the original paper
        qr = einsum("b p n d, 1 p m d -> b p n m", q, relative_pos)*self.scale
        qk = einsum("b p n d, b p m d -> b p n m", q, k)*self.scale

        attention_weight = (qr + qk).softmax(dim=-1)
        out = einsum("b p n m, b p n d -> b p n d", attention_weight, v)
        out = rearrange(out, "b p (h w) d -> b (p d) h w")

        return out


class BoTBlock(nn.Module):
    def __init__(self, *, in_channel, out_channel, heads, head_dim=None, **kwargs):
        super().__init__()
    
