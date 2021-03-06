from abc import abstractmethod

import torch
from torch import nn
from torchvision import transforms


class UNetConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UNetConvBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class UNetEncoder(nn.Module):
    def __init__(self, input_channel=1, channel_in_between=[]):
        super(UNetEncoder, self).__init__()
        
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
    def __init__(self, middle_channel=1024, channel_in_between=[]):
        super(UNetDecoder, self).__init__()

        assert len(channel_in_between) >= 1, f"[{self.__class__.__name__}] Please specify the number of channels for at least 1 layer."

        channel_in_between = [middle_channel] + channel_in_between
        self.layers = nn.ModuleList([
            nn.ModuleList([
                UNetConvBlock(channel_in_between[idx], channel_in_between[idx + 1]),
                nn.ConvTranspose2d(channel_in_between[idx], channel_in_between[idx + 1], kernel_size=2, stride=2)
            ]) 
            for idx in range(len(channel_in_between) - 1)
        ])

    def forward(self, x, hidden_xs):
        for (convBlock, upSampling), hidden_x in zip(self.layers, hidden_xs):
            x = convBlock(x)
            hidden_x = self.crop(hidden_x, x.shape)
            x = torch.cat([hidden_x, x], dim=1)
            x = upSampling(x)

        return x

    def crop(self, hidden_x, shape):
        _, _, h, w = shape

        return transforms.functional.center_crop(hidden_x, [h, w])


class UNetBase(nn.Module):
    def __init__(self, *, input_channel=1, middle_channel=1024, output_channel=1, channel_in_between=[], to_remain_size=False, image_size=None):
        super(UNetBase, self).__init__()

        assert len(channel_in_between) >= 1, f"[{self.__class__.__name__}] Please specify the number of channels for at least 1 layer."

        self.to_remain_size = to_remain_size
        if to_remain_size:
            self.image_size = image_size

        self.encoder = UNetEncoder(input_channel, channel_in_between)
        self.middle_layer = self._build_middle_layer(channel_in_between[-1], middle_channel)
        self.decoder = UNetDecoder(middle_channel, channel_in_between[::-1])
        self.output_layer = nn.Conv2d(channel_in_between[0], output_channel, kernel_size=1)

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
        
    @abstractmethod
    def _build_middle_layer(self, in_channel, out_channel):
        ...


class UNet(UNetBase):
    def _build_middle_layer(self, in_channel, out_channel):
        return UNetConvBlock(in_channel, out_channel)


if __name__ == "__main__":

    unet = UNet(
        input_channel=3,
        middle_channel=1024,
        output_channel=10,
        channel_in_between=[64, 128, 256, 512],
        to_remain_size=True
    )
    print(unet)

    x = torch.randn(1, 3, 572, 572)

    print(unet(x).shape)