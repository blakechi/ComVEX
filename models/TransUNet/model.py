import torch
from torch import nn

from models.transformer import Transformer
from models.utils import UNetBase


class TransUNet(UNetBase):
    def _build_middle_layer(self, in_channel, out_channel):
        ...


if __name__ == "__main__":
    transUnet = TransUNet(
        input_channel=3,
        middle_channel=1024,
        output_channel=10,
        channel_in_between=[64, 128, 256, 512],
        to_remain_size=True
    )
    print(transUnet)

    x = torch.randn(1, 3, 572, 572)

    print(transUnet(x).shape)