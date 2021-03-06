import torch
from torch import nn


class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        stride=None,
        padding=None,
        norm_layer=None,
        down_sampling=None
        ):
        super().__init__()

        if stride is not None:
            self.stride = stride
        else:
            self.stride = 1 if in_channels == out_channels else 2

        self.padding = padding if padding is not None else 1
        self.norm = nn.BatchNorm2d if norm_layer is None else norm_layer
        self.down_sampling = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, self.stride),
            self.norm(out_channels)
        ) if down_sampling is not None else None

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                3, 
                self.stride, 
                self.padding
            ),
            self.norm(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, 
                out_channels, 
                3, 
                self.stride, 
                self.padding
            ),
            self.norm(num_features=out_channels)
        )
        self.relu = nn.ReLU(inplace=True)  # after addition

    def forward(self, x):
        if self.down_sampling is not None:
            return self.relu(self.layers(x) + self.down_sampling(x))
        
        return self.relu(self.layers(x) + x)


class ResNetBottleneckBlock(ResNetBlock):
    def __init__(self, in_channel, out_channel, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                in_channels, 
                1, 
                self.stride, 
                self.padding
            ),
            self.norm(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, 
                in_channels, 
                3, 
                self.stride, 
                self.padding
            ),
            self.norm(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, 
                out_channels, 
                1, 
                self.stride, 
                self.padding
            ),
            self.norm(num_features=out_channels)
        )
        self.relu = nn.ReLU(inplace=True)  # after addition
        self.down_sampling = down_sampling


class FullPreActivationResNetBlock(ResNetBlock):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)

        self.layers = nn.Sequential(
            self.norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                self.stride,
                self.padding
            ),
            self.norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                self.stride,
                self.padding
            ),
        )

    def forward(self, x):
        if self.down_sampling is not None:
            return self.layers(x) + self.down_sampling(x)
        
        return self.layers(x) + x

if __name__ == "__main__":
    FullPreActivationResNetBlock(10, 10)