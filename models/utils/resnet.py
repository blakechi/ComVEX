import torch
from torch import nn


class ResNetBlockBase(nn.Module):
    """
    For storing important coefficients arocss different type of ResNet.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        stride=None,
        padding=None,
        norm_layer=None,
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
        ) if self.stride != 1 else None


class ResNetBlock(ResNetBlockBase):
    def __init__(
        self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)

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
    def __init__(self, in_channel, out_channel, **kwargs):
        super().__init__(in_channel, out_channel, **kwargs)

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


class FullPreActivationResNet(ResNetBlockBase):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)

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


class FullPreActivationResNetBottleneck(FullPreActivationResNet):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)

        self.layers = nn.Sequential(
            self.norm(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                in_channels,
                1,
                1,
                0
            ),
            self.norm(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                in_channels,
                3,
                self.stride,
                self.padding
            ),
            self.norm(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                1,
                1,
                0
            ),
        )


if __name__ == "__main__":
    b = FullPreActivationResNetBottleneck(10, 10)

    a = torch.rand(1, 10, 32, 32)

    print(b(a).size())