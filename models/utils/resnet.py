import torch
from torch import nn


# https://arxiv.org/pdf/1512.03385.pdf
class ResNetBlockBase(nn.Module):
    """
    For storing important coefficients arocss different type of ResNet.
    """
    def __init__(
        self,
        in_channel,
        out_channel,
        *,
        stride=None,
        padding=None,
        norm_layer=None,
        remain_dim=False  # False when want to increase the number of channels but remain width and height
    ):
        super().__init__()

        if stride is not None:
            self.stride = stride
        else:
            self.stride = 1
            if (in_channel != out_channel) and not remain_dim:
                self.stride = 2

        self.padding = padding if padding is not None else 1
        self.norm = norm_layer if norm_layer is not None else nn.BatchNorm2d

        if remain_dim:
            self.skip = nn.Conv2d(in_channel, out_channel, 1)
        else:
            self.skip = nn.Conv2d(in_channel, out_channel, 1, self.stride) if self.stride > 1 else nn.Identity()


class ResNetBlock(ResNetBlockBase):
    def __init__(
        self, in_channel, out_channel, **kwargs):
        super().__init__(in_channel, out_channel, **kwargs)

        self._net = nn.Sequential(
            nn.Conv2d(
                in_channel, 
                out_channel, 
                3, 
                self.stride, 
                self.padding
            ),
            self.norm(num_features=out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channel, 
                out_channel, 
                3,
                1,
                1
            ),
            self.norm(num_features=out_channel)
        )
        self.relu = nn.ReLU(inplace=True)  # after addition

    def forward(self, x):
        return self.relu(self._net(x) + self.skip(x))


class ResNetBottleneckBlock(ResNetBlock):
    def __init__(self, in_channel, out_channel, **kwargs):
        super().__init__(in_channel, out_channel, **kwargs)

        self._net = nn.Sequential(
            nn.Conv2d(
                in_channel, 
                in_channel, 
                1, 
            ),
            self.norm(num_features=in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channel, 
                in_channel, 
                3, 
                self.stride, 
                self.padding
            ),
            self.norm(num_features=in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channel, 
                out_channel, 
                1, 
            ),
            self.norm(num_features=out_channel)
        )
        self.relu = nn.ReLU(inplace=True)  # after addition


# https://arxiv.org/pdf/1603.05027.pdf
class ResNetFullPreActivationBlock(ResNetBlockBase):
    def __init__(self, in_channel, out_channel, **kwargs):
        super().__init__(in_channel, out_channel, **kwargs)

        self._net = nn.Sequential(
            self.norm(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channel,
                out_channel,
                3,
                self.stride,
                self.padding
            ),
            self.norm(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channel,
                out_channel,
                3,
                1,
                1,
            ),
        )

    def forward(self, x):
        return self._net(x) + self.skip(x)


class ResNetFullPreActivationBottleneckBlock(ResNetFullPreActivationBlock):
    def __init__(self, in_channel, out_channel, **kwargs):
        super().__init__(in_channel, out_channel, **kwargs)

        self._net = nn.Sequential(
            self.norm(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channel,
                in_channel,
                1,
            ),
            self.norm(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channel,
                in_channel,
                3,
                self.stride,
                self.padding
            ),
            self.norm(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channel,
                out_channel,
                1,
            ),
        )


if __name__ == "__main__":
    # b = ResNetBlock(10, 10, stride=2, padding=1)
    # b = ResNetBottleneckBlock(10, 10, stride=2, padding=1)
    # b = ResNetFullPreActivationBlock(10, 10, stride=2, padding=1)
    b = ResNetFullPreActivationBottleneckBlock(3, 10)

    a = torch.rand(1, 3, 32, 32)

    print(b(a).size())