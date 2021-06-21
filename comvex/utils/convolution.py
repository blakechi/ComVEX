from typing import Optional

from torch import nn

from comvex.utils.helpers import name_with_msg


class XXXConvXdBase(nn.Module):
    __constants__ = ["in_channel", "out_channel"]

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

        if dimension == 1:
            self.conv = nn.Conv1d
        elif dimension == 2:
            self.conv = nn.Conv2d
        else: 
            self.conv = nn.Conv3d
    