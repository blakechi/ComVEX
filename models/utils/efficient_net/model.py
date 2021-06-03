from typing import Optional

import torch
from torch import nn


class SeperateConv2D(nn.Module):
    def __init__(
        self, 
        in_channel: int, 
        out_channel: int, 
        kernel_size: int = 3, 
        padding: int = 1, 
        kernels_per_layer: int = 1, 
        norm_layer: Optional[str] = None,
        act_fnc: Optional[str] = None,
        **kwargs
    ):
        super().__init__()

        self.depth_wise_conv = nn.Conv2d(
            in_channel, 
            in_channel*kernels_per_layer, 
            kernel_size, 
            padding=padding, 
            groups=in_channel
        )
        self.norm_0 = getattr(norm_layer)(in_channel*kernels_per_layer, **kwargs) if hasattr(nn, norm_layer) else None
        self.act_fnc = getattr(nn, act_fnc) if hasattr(nn, act_fnc) else None
        
        self.pixel_wise_conv = nn.Conv2d(
            in_channel*kernels_per_layer,
            out_channel,
            kernel_size=1,
        )
        self.norm_1 = getattr(norm_layer)(in_channel*kernels_per_layer, **kwargs) if hasattr(nn, norm_layer) else None

    def forward(self, x):
        x = self.depth_wise_conv(x)
        if self.norm_0:
            x = self.norm_0(x)
        if self.act_fnc:
            x = self.act_fnc(x)

        x = self.pixel_wise_conv(x)
        if self.norm_1:
            x = self.norm_1(x)
            
        return x
