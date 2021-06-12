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
        norm_layer_name: Optional[str] = None,
        act_fnc_name: Optional[str] = None,
        dimension: int = 2,
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

        self.norm_0 = getattr(nn, norm_layer_name)(
            in_channel*kernels_per_layer,
            **kwargs
        ) if norm_layer_name and hasattr(nn, norm_layer_name) else None
        self.norm_1 = getattr(nn, norm_layer_name)(
            in_channel*kernels_per_layer,
            **kwargs
        ) if norm_layer_name and hasattr(nn, norm_layer_name) else None
            
        self.act_fnc = getattr(nn, act_fnc_name)() if act_fnc_name and hasattr(nn, act_fnc_name) else None

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


class MBConvXd(nn.Module):
    r"""
    Reference from: 
        1. MobileNetV2 (https://arxiv.org/pdf/1801.04381.pdf)
        2. MnasNet (https://arxiv.org/pdf/1807.11626.pdf)
    """
    def __init__(
        self, 
        in_channel: int, 
        out_channel: int, 
        expand_channel: Optional[int] = None,
        expand_scale: Optional[int] = None, 
        kernel_size: int = 3, 
        padding: int = 1, 
        first_pixel_wise_conv_stride: int = 1,
        norm_layer_name: Optional[str] = None,
        act_fnc_name: Optional[str] = None,
        dimension: int = 2,
        **kwargs  # For example: `eps` and `elementwise_affine` for `nn.LayerNorm`
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

        assert (
            expand_channel is not None or expand_scale is not None
        ), name_with_msg(self, "Either `expand_channel` or `expand_scale` should be specified")
        expand_channel = expand_channel if expand_channel is not None else in_channel*expand_scale

        self.pixel_wise_conv_0 = conv(
            in_channel,
            expand_channel,
            kernel_size=1,
            stride=first_pixel_wise_conv_stride,
        )
        self.depth_wise_conv = conv(
            expand_channel, 
            expand_channel, 
            kernel_size, 
            padding=padding, 
            groups=expand_channel
        )
        self.pixel_wise_conv_1 = conv(
            expand_channel,
            out_channel,
            kernel_size=1,
        )

        self.pixel_wise_norm_0 = getattr(nn, norm_layer_name)(
            expand_channel,
            **kwargs
        ) if norm_layer_name and hasattr(nn, norm_layer_name) else None
        self.depth_wise_norm = getattr(nn, norm_layer_name)(
            expand_channel,
            **kwargs
        ) if norm_layer_name and hasattr(nn, norm_layer_name) else None
        self.pixel_wise_norm_1 = getattr(nn, norm_layer_name)(
            out_channel,
            **kwargs
        ) if norm_layer_name and hasattr(nn, norm_layer_name) else None
            
        self.pixel_wise_act_fnc = getattr(nn, act_fnc_name)() if act_fnc_name and hasattr(nn, act_fnc_name) else None
        self.depth_wise_act_fnc = getattr(nn, act_fnc_name)() if act_fnc_name and hasattr(nn, act_fnc_name) else None

    def forward(self, x):

        # First pixel-wise conv
        x = self.pixel_wise_conv_0(x)
        if self.pixel_wise_norm_0:
            x = self.pixel_wise_norm_0(x)
        if self.pixel_wise_act_fnc:
            x = self.pixel_wise_act_fnc(x)
        
        # Depth-wise conv
        x = self.depth_wise_conv(x)
        if self.depth_wise_norm:
            x = self.depth_wise_norm(x)
        if self.depth_wise_act_fnc:
            x = self.depth_wise_act_fnc(x)

        # Second pixel-wise conv
        x = self.pixel_wise_conv_1(x)
        if self.pixel_wise_norm_1:
            x = self.pixel_wise_norm_1(x)
            
        return x

    @classmethod
    def MBConv2d6k3(cls, in_channel: int, out_channel: int) -> "MBConvXd":
        r"""
        The MBConv6 (k3x3) from MnasNet (https://arxiv.org/pdf/1807.11626.pdf)
        """
        return cls(
            in_channel,
            out_channel,
            expand_scale=6,
            kernel_size=3,
            norm_layer="BatchNorm2d",
            act_fnc_name="ReLU"
        )