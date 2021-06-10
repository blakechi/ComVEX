import torch
from torch import nn
from torchvision import transforms


class UNetConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class UNetBilinearUpsamplingBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.net = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1),
        )

    def forward(self, x):
        return self.net(x)


class UNetEncoder(nn.Module):
    def __init__(self, input_channel=1, channel_in_between=[]):
        super().__init__()
        
        assert len(channel_in_between) >= 1, f"[{self.__class__.__name__}] Please specify the number of channels for at least 1 layer."

        channel_in_between = [input_channel] + channel_in_between
        self.layers = nn.ModuleList([
            nn.ModuleList([
                UNetConvBlock(channel_in_between[idx], channel_in_between[idx + 1]),
                nn.MaxPool2d(kernel_size=2)
            ]) 
            for idx in range(len(channel_in_between) - 1)
        ])

    def forward(self, x):
        hidden_xs = []
        for convBlock, downSampling in self.layers:
            x = convBlock(x)
            hidden_xs.append(x)
            x = downSampling(x)

        return x, hidden_xs


class UNetDecoder(nn.Module):
    def __init__(self, middle_channel=1024, channel_in_between=[], use_bilinear_upsampling=False):
        super().__init__()

        assert len(channel_in_between) >= 1, f"[{self.__class__.__name__}] Please specify the number of channels for at least 1 layer."

        channel_in_between = [middle_channel] + channel_in_between
        self.layers = nn.ModuleList([
            nn.ModuleList([
                UNetConvBlock(channel_in_between[idx], channel_in_between[idx + 1]),
                nn.ConvTranspose2d(channel_in_between[idx], channel_in_between[idx + 1], kernel_size=2, stride=2) if not use_bilinear_upsampling else UNetBilinearUpsamplingBlock(channel_in_between[idx], channel_in_between[idx + 1])
            ]) 
            for idx in range(len(channel_in_between) - 1)
        ])

    def forward(self, x, hidden_xs):
        for (convBlock, upSampling), hidden_x in zip(self.layers, hidden_xs):
            x = upSampling(x)
            hidden_x = self.crop(hidden_x, x.shape)
            x = torch.cat([hidden_x, x], dim=1)
            x = convBlock(x)

        return x

    def crop(self, hidden_x, shape):
        _, _, h, w = shape

        return transforms.functional.center_crop(hidden_x, [h, w])


class UNetBase(nn.Module):
    def __init__(self, *, channel_in_between=[], to_remain_size=False, image_size=None):
        super().__init__()

        assert isinstance(channel_in_between, list) or len(channel_in_between) >= 1, f"[{self.__class__.__name__}] Please specify the number of channels for at least 1 layer."

        self.channel_in_between = channel_in_between
        self.to_remain_size = to_remain_size
        if to_remain_size:
            assert image_size is not None, f"[{self.__class__.__name__}] Please specify the image size to remain output size as the input."
            self.image_size = image_size 


class UNet(UNetBase):
    """
    Architecture:
        encoder               decoder --> output_layer
           |                     ^ 
           |                     |
             ->  middle_layer --
    """
    def __init__(
        self,
        input_channel=1, 
        middle_channel=1024, 
        output_channel=1, 
        **kwargs
        ):
        super().__init__(**kwargs)

        self.encoder = UNetEncoder(input_channel, self.channel_in_between)
        self.middle_layer = UNetConvBlock(self.channel_in_between[-1], middle_channel)
        self.decoder = UNetDecoder(middle_channel, self.channel_in_between[::-1])
        self.output_layer = nn.Conv2d(self.channel_in_between[0], output_channel, kernel_size=1)

    def forward(self, x):
        _, _, h, w = x.shape

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