from comvex.utils.helpers.functions import get_attr_if_exists
from typing import Optional, Tuple, Dict

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
    