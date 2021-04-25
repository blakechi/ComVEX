import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from models.utils import ResNetBlockBase


class BoTNetMHSA(nn.Module):
    def __init__(self, *, lateral_size, dim, heads, head_dim=None, **kwargs):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim if head_dim is not None else dim // heads

        assert (
            self.head_dim * self.heads == self.dim
        ), "Head dimension times the number of heads must be equal to embedding dimension"

        self.QKV = nn.Conv2d(dim, 3*dim, 1, bias=False)
        self.height_relative_pos = nn.Parameter(torch.empty(lateral_size, 1, dim))
        self.width_relative_pos = nn.Parameter(torch.empty(1, lateral_size, dim))
        nn.init.kaiming_uniform_(self.height_relative_pos)
        nn.init.kaiming_uniform_(self.width_relative_pos)

    def forward(self, x):
        b, e, h, w, p, device = *x.shape, self.heads, x.device

        q, k, v = self.QKV(x).chunk(chunks=3, dim=-3)
        q, k, v = map(lambda t: rearrange(t, "b (p d) h w -> b p (h w) d", p=p), (q, k, v))

        relative_pos = rearrange(self.height_relative_pos + self.width_relative_pos, "h w (p d) -> 1 p (h w) d", p=p)

        # Note: scaling doesn't be mentioned in the original paper
        qr = einsum("b p n d, l p m d -> b p n m", q, relative_pos)  # l means 1
        qk = einsum("b p n d, b p m d -> b p n m", q, k)

        similarity = (qr + qk).softmax(dim=-1)
        out = einsum("b p n m, b p m d -> b p n d", similarity, v)
        out = rearrange(out, "b p (h w) d -> b (p d) h w", h=h)

        return out


class BoTNetBlock(ResNetBlockBase):
    def __init__(
        self, 
        in_channel, 
        out_channel, 
        *, 
        lateral_size,
        heads, 
        **kwargs
    ):
        super().__init__(in_channel, out_channel, **kwargs)

        _net = [
            nn.Conv2d(
                in_channel, 
                in_channel, 
                1
            ),
            self.Norm(num_features=in_channel),
            nn.ReLU(inplace=True),
            BoTNetMHSA(
                lateral_size=lateral_size,
                dim=in_channel,
                heads=heads, 
                **kwargs
            ),
            self.Norm(num_features=in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channel, 
                out_channel, 
                1
            ),
            self.Norm(num_features=out_channel),
        ]
        if 'stride' in kwargs: _net.insert(-4, nn.AvgPool2d(kernel_size=2, stride=2))

        self._net = nn.Sequential(*_net)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self._net(x) + self.skip(x))


class BoTNetBlockFullPreActivation(ResNetBlockBase):
    def __init__(
        self, 
        in_channel, 
        out_channel, 
        *, 
        lateral_size,
        heads, 
        **kwargs
    ):
        super().__init__(in_channel, out_channel, **kwargs)

        _net = [
            self.Norm(num_features=in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channel, 
                in_channel, 
                1
            ),
            self.Norm(num_features=in_channel),
            nn.ReLU(inplace=True),
            BoTNetMHSA(
                lateral_size=lateral_size,
                dim=in_channel,
                heads=heads, 
                **kwargs
            ),
            self.Norm(num_features=in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channel, 
                out_channel, 
                1
            ),
        ]
        if 'stride' in kwargs: _net.insert(-3, nn.AvgPool2d(kernel_size=2, stride=2))

        self._net = nn.Sequential(*_net)

    def forward(self, x):
        return self._net(x) + self.skip(x)


class BoTNet(nn.Module):
    def __init__(self, ):
        super().__init__()