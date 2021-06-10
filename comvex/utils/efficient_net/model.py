from typing import Optional

import torch
from torch import nn

from comvex.utils.helpers import name_with_msg

class SeperateConvXd(nn.Module):
    def __init__(
        self, 
        in_channel: int, 
        out_channel: int, 
        kernel_size: int = 3, 
        padding: int = 1, 
        kernels_per_layer: int = 1, 
        dimension: int = 2,
        norm_layer: Optional[str] = None,
        act_fnc: Optional[str] = None,
        **kwargs
    ):
        super().__init__()

        assert (
            (0 < dimension) and (dimension < 4)
        ), name_with_msg(self, "`dimension` must be larger than 0 and smaller than 4")

        if dimension == 1:
            conv = nn.Conv1d
        elif dimension == 2:
            conv = nn.Conv2d
        else: 
            conv = nn.Conv3d

        self.depth_wise_conv = conv(
            in_channel, 
            in_channel*kernels_per_layer, 
            kernel_size, 
            padding=padding, 
            groups=in_channel
        )
        self.pixel_wise_conv = conv(
            in_channel*kernels_per_layer,
            out_channel,
            kernel_size=1,
        )

        if norm_layer:
            self.norm_0 = getattr(norm_layer)(in_channel*kernels_per_layer, **kwargs) if hasattr(nn, norm_layer) else None
            self.norm_1 = getattr(norm_layer)(in_channel*kernels_per_layer, **kwargs) if hasattr(nn, norm_layer) else None
        else:
            self.norm_0, self.norm_1 = None, None
            
        if act_fnc:
            self.act_fnc = getattr(nn, act_fnc) if hasattr(nn, act_fnc) else None
        else:
            self.act_fnc = None

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