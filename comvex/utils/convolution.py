from typing import Optional

from torch import nn
try:
    from typing_extensions import Final
except:
    from torch.jit import Final

from comvex.utils.helpers import name_with_msg


class XXXConvXdBase(nn.Module):
    r"""
    A simple wrapper for determining the number of input/output channels
    and which types (1, 2, 3D) of convolution layer would be used.
    """

    in_channel: Final[int]
    out_channel: Final[int]
    dimension: Final[int]

    def __init__(
        self,
        in_channel: int,
        out_channel: Optional[int] = None,
        dimension: int = 2,
    ) -> None:
        super().__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel if out_channel is not None else in_channel

        assert (
            (0 < dimension) and (dimension < 4)
        ), name_with_msg(self, "`dimension` must be larger than 0 and smaller than 4")
        
        self.dimension = dimension

        if self.dimension == 1:
            self.conv = nn.Conv1d
        elif self.dimension == 2:
            self.conv = nn.Conv2d
        else: 
            self.conv = nn.Conv3d
    