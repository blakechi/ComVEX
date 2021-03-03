import torch
from torch import nn
from torchvision import transforms


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, input_channel=1, channel_in_between=[]):
        super(Encoder, self).__init__()
        
        assert len(channel_in_between) >= 1, f"[{self.__class__.__name__}] Please specify the number of channels for at least 1 layer."

        channel_in_between = [input_channel] + channel_in_between
        self.layers = nn.ModuleList([
            nn.ModuleList([
                ConvBlock(channel_in_between[idx], channel_in_between[idx + 1]),
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


class Decoder(nn.Module):
    def __init__(self, middle_channel=1024, channel_in_between=[]):
        super(Decoder, self).__init__()

        assert len(channel_in_between) >= 1, f"[{self.__class__.__name__}] Please specify the number of channels for at least 1 layer."

        channel_in_between = [middle_channel] + channel_in_between
        self.layers = nn.ModuleList([
            nn.ModuleList([
                ConvBlock(channel_in_between[idx], channel_in_between[idx + 1]),
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


class UNet(nn.Module):
    def __init__(self, *, input_channel=1, middle_channel=1024, output_channel=1, channel_in_between=[], to_remain_size=False, image_size=None):
        super(UNet, self).__init__()

        assert len(channel_in_between) >= 1, f"[{self.__class__.__name__}] Please specify the number of channels for at least 1 layer."

        self.to_remain_size = to_remain_size
        if to_remain_size:
            self.image_size = image_size

        self.encoder = Encoder(input_channel, channel_in_between)
        self.middle_layer = ConvBlock(channel_in_between[-1], middle_channel)
        self.decoder = Decoder(middle_channel, channel_in_between[::-1])
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