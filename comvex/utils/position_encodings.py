import math

import torch
from torch import nn


class PositionEncodingFourier(nn.Module):
    """
    Copy from: https://github.com/facebookresearch/xcit/blob/master/xcit.py#L20
    """

    def __init__(self, hidden_dim=32, dim=768, temperature=10000, to_flatten=True):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim
        self.to_flatten = to_flatten

    def forward(self, B: int, H: int, W: int):
        mask = torch.zeros(B, H, W, dtype=torch.bool).to(self.token_projection.weight.device)
        not_mask = ~mask

        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = x_embed[:, :, :, None] / dim_t

        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)

        return pos.view(B, -1, self.dim) if self.to_flatten else pos