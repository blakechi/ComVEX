from torch import nn
from einops.layers.torch import Rearrange

from comvex.transunet.utils import TransUNetEncoderConvBlock, TransUNetViT
from comvex.utils import UNetBase, UNetDecoder


class TransUNetEncoder(nn.Module):
    def __init__(self, input_channel, channel_in_between, num_res_blocks_in_between, vit_input_size, **kwargs):
        super().__init__()
        
        assert len(channel_in_between) >= 1, f"[{self.__class__.__name__}] Please specify the number of channels for at least 1 layer."

        channel_in_between = [input_channel] + channel_in_between
        self.layers = nn.ModuleList([
            TransUNetEncoderConvBlock(channel_in_between[idx], channel_in_between[idx + 1], num_res_blocks_in_between[idx])
            for idx in range(len(channel_in_between) - 1)
        ])
        self.vit = TransUNetViT(
            image_size=vit_input_size,
            image_channel=channel_in_between[-1],
            **kwargs
        )

    def forward(self, x):
        hidden_xs = []
        for convBlock in self.layers:
            x = convBlock(x)
            hidden_xs.append(x)

        x = self.vit(x)

        return x, hidden_xs


class TransUNet(UNetBase):
    """
    Architecture:
        encoder               decoder --> output_layer
           |       .......       ^ 
           |                     |
             ->  middle_layer --
    """
    def __init__(
        self,
        input_channel, 
        middle_channel, 
        output_channel, 
        channel_in_between,
        num_res_blocks_in_between,
        image_size,
        to_remain_size=False,
        **kwargs
    ):
        super().__init__(channel_in_between=channel_in_between, to_remain_size=to_remain_size, image_size=image_size)

        vit_input_size = self.image_size // 2**len(self.channel_in_between)
        vit_output_size = vit_input_size // 2
        self.encoder = TransUNetEncoder(
            input_channel, 
            self.channel_in_between,
            num_res_blocks_in_between,
            vit_input_size,
            **kwargs
        )
        self.middle_layer = Rearrange("b (p q) d -> b d p q", p=vit_output_size)
        self.decoder = UNetDecoder(middle_channel, self.channel_in_between[::-1], use_bilinear_upsampling=True)
        self.output_layer = nn.Conv2d(self.channel_in_between[0], output_channel, kernel_size=1)  # kernel_size == 3 in the offical code

    def forward(self, x):
        b, c, h, w = x.shape

        x, hidden_xs = self.encoder(x)
        x = self.middle_layer(x)
        x = self.decoder(x, hidden_xs[::-1])
        x = self.output_layer(x)
        
        if self.to_remain_size:
            x = nn.functional.interpolate(
                x, 
                self.image_size if self.image_size is not None else (h, w)
            )
            
        return x