import torch
from torch import nn


class ResNetBlockBase(nn.Module):
    """
    For storing important coefficients arocss different type of ResNet.
    """
    def __init__(
        self,
        input_channel,
        output_channel,
        *,
        stride=None,
        padding=None,
        norm_layer=None,
    ):
        super().__init__()

        if stride is not None:
            self.stride = stride
        else:
            self.stride = 1 if input_channel == output_channel else 2

        self.padding = padding if padding is not None else 1
        self.norm = nn.BatchNorm2d if norm_layer is None else norm_layer
        self.down_sampling = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 1, self.stride),
            self.norm(output_channel)
        ) if self.stride != 1 else None


class ResNetBlock(ResNetBlockBase):
    def __init__(
        self, input_channel, output_channel, **kwargs):
        super().__init__(input_channel, output_channel, **kwargs)

        self.layers = nn.Sequential(
            nn.Conv2d(
                input_channel, 
                output_channel, 
                3, 
                self.stride, 
                self.padding
            ),
            self.norm(num_features=output_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channel, 
                output_channel, 
                3, 
                self.stride, 
                self.padding
            ),
            self.norm(num_features=output_channel)
        )
        self.relu = nn.ReLU(inplace=True)  # after addition

    def forward(self, x):
        if self.down_sampling is not None:
            return self.relu(self.layers(x) + self.down_sampling(x))
        
        return self.relu(self.layers(x) + x)


class ResNetBottleneck(ResNetBlock):
    def __init__(self, in_channel, out_channel, **kwargs):
        super().__init__(in_channel, out_channel, **kwargs)

        self.layers = nn.Sequential(
            nn.Conv2d(
                input_channel, 
                input_channel, 
                1, 
                self.stride, 
                self.padding
            ),
            self.norm(num_features=input_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                input_channel, 
                input_channel, 
                3, 
                self.stride, 
                self.padding
            ),
            self.norm(num_features=input_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                input_channel, 
                output_channel, 
                1, 
                self.stride, 
                self.padding
            ),
            self.norm(num_features=output_channel)
        )
        self.relu = nn.ReLU(inplace=True)  # after addition


class ResNetFullPreActivation(ResNetBlockBase):
    def __init__(self, input_channel, output_channel, **kwargs):
        super().__init__(input_channel, output_channel, **kwargs)

        self.layers = nn.Sequential(
            self.norm(output_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                input_channel,
                output_channel,
                3,
                self.stride,
                self.padding
            ),
            self.norm(output_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                input_channel,
                output_channel,
                3,
                self.stride,
                self.padding
            ),
        )

    def forward(self, x):
        if self.down_sampling is not None:
            return self.layers(x) + self.down_sampling(x)
        
        return self.layers(x) + x


class ResNetFullPreActivationBottleneck(ResNetFullPreActivation):
    def __init__(self, input_channel, output_channel, **kwargs):
        super().__init__(input_channel, output_channel, **kwargs)

        self.layers = nn.Sequential(
            self.norm(input_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                input_channel,
                input_channel,
                1,
                1,
                0
            ),
            self.norm(input_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                input_channel,
                input_channel,
                3,
                self.stride,
                self.padding
            ),
            self.norm(input_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                input_channel,
                output_channel,
                1,
                1,
                0
            ),
        )


if __name__ == "__main__":
    b = ResNetFullPreActivationBottleneck(10, 10)

    a = torch.rand(1, 10, 32, 32)

    print(b(a).size())