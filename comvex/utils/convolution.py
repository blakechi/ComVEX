from comvex.utils.helpers.functions import get_attr_if_exists
from typing import Optional, Tuple, Dict

import torch
from torch import nn
try:
    from typing_extensions import Final
except:
    from torch.jit import Final

from comvex.utils.helpers import name_with_msg, get_attr_if_exists


class XXXConvXdBase(nn.Module):
    r"""
    A simple wrapper for determining the number of input/output channels
    and which types (1, 2, 3D) of convolution layer would be used.

    Can get any official `nn.Module` whose name ends with `Xd`, where `X` is 1, 2, or 3, from PyTorch.
    """

    in_channel: Final[int]
    out_channel: Final[int]
    dimension: Final[int]
    default_components: Final[Dict[str, str]] = { "conv": "Conv" }

    def __init__(
        self,
        in_channel: int,
        out_channel: Optional[int] = None,
        dimension: int = 2,
        *,
        extra_components: Dict[str, str] = {}
    ) -> None:
        super().__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel if out_channel is not None else in_channel

        assert (
            (0 < dimension) and (dimension < 4)
        ), name_with_msg(self, "`dimension` must be larger than 0 and smaller than 4")
        
        self.dimension = dimension
        component_orders = { **self.default_components, **extra_components}
        components = self._get_component(component_orders)
        for attr_name, module in components:
            setattr(self, attr_name, module)
        
    def _get_component(self, component_orders: Dict[str, str]) -> Dict[str, nn.Module]:
        modules = [get_attr_if_exists(nn, f"{component}{self.dimension}d") for component in component_orders.values()]
        
        return dict([(key, value) for key, value in zip(component_orders.keys(), modules)])
    

class SeperableConvXd(XXXConvXdBase):
    r"""
    Reference from: MnasNet (https://arxiv.org/pdf/1807.11626.pdf)
    """

    def __init__(
        self, 
        in_channel: int, 
        out_channel: int, 
        kernel_size: int = 3, 
        padding: int = 1, 
        kernels_per_layer: int = 1, 
        dimension: int = 2,
        act_fnc_name: str = "ReLU",
        extra_components: Dict[str, str] = {"norm", "BatchNorm"},
        use_conv_only: bool = False,
        **kwargs  # For the normalization layer
    ) -> None:
        super().__init__(in_channel, out_channel, dimension=dimension, extra_components=extra_components)

        self.depth_wise_conv = self.conv(
            self.in_channel, 
            self.in_channel*kernels_per_layer, 
            kernel_size, 
            padding=padding, 
            groups=self.in_channel
        )
        self.pixel_wise_conv = self.conv(
            self.in_channel*kernels_per_layer,
            out_channel,
            kernel_size=1,
        )

        self.use_conv_only = use_conv_only
        if not self.use_conv_only:
            self.norm_layer = self.norm(
                self.in_channel*kernels_per_layer,
                **kwargs
            )
            self.act_fnc = get_attr_if_exists(nn, act_fnc_name)()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_wise_conv(x)
        x = self.pixel_wise_conv(x)
        
        if self.use_conv_only:
            x = self.norm_layer(x)
            x = self.act_fnc(x)
        
        return x