from typing import Union, Optional
from functools import partial
from collections import OrderedDict

import torch
from torch import nn
from einops.layers.torch import Rearrange

from .config import ResNetConfig


ReLU = partial(nn.ReLU, inplace=True)


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
        remain_dim=False  # True when want to increase the number of channels but remain width and height
    ):
        super().__init__()

        if stride is not None:
            self.stride = stride
        else:
            self.stride = 1
            if (in_channel != out_channel) and not remain_dim:
                self.stride = 2

        self.padding = padding if padding is not None else 1
        self.Norm = norm_layer if norm_layer is not None else nn.BatchNorm2d

        if remain_dim:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1),
                self.Norm(out_channel)
            )
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, self.stride),
                self.Norm(out_channel)
            ) if self.stride > 1 else nn.Identity()


class ResNetBlock(ResNetBlockBase):
    def __init__(self, in_channel, out_channel, **kwargs):
        super().__init__(in_channel, out_channel, **kwargs)

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channel, 
                out_channel, 
                3, 
                self.stride, 
                self.padding
            ),
            self.Norm(num_features=out_channel),
            ReLU(),
            nn.Conv2d(
                out_channel, 
                out_channel, 
                3,
                1,
                1
            ),
            self.Norm(num_features=out_channel)
        )
        self.relu = ReLU()  # after addition

    def forward(self, x):
        return self.relu(self.net(x) + self.skip(x))


class ResNetBottleneckBlock(ResNetBlockBase):
    def __init__(self, in_channel, base_channel, expand_channel=None, expand_scale=4, **kwargs):
        expand_channel = expand_channel if expand_channel is not None else expand_scale*base_channel
        super().__init__(in_channel, expand_channel, **kwargs)

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channel, 
                base_channel, 
                1, 
            ),
            self.Norm(num_features=base_channel),
            ReLU(),
            nn.Conv2d(
                base_channel, 
                base_channel, 
                3, 
                self.stride, 
                self.padding
            ),
            self.Norm(num_features=base_channel),
            ReLU(),
            nn.Conv2d(
                base_channel, 
                expand_channel, 
                1, 
            ),
            self.Norm(num_features=expand_channel)
        )
        self.relu = ReLU()  # after addition

    def forward(self, x):
        return self.relu(self.net(x) + self.skip(x))


# https://arxiv.org/pdf/1603.05027.pdf
class ResNetFullPreActivationBlock(ResNetBlockBase):
    def __init__(self, in_channel, out_channel, **kwargs):
        super().__init__(in_channel, out_channel, **kwargs)

        self.net = nn.Sequential(
            self.Norm(out_channel),
            ReLU(),
            nn.Conv2d(
                in_channel,
                out_channel,
                3,
                self.stride,
                self.padding
            ),
            self.Norm(out_channel),
            ReLU(),
            nn.Conv2d(
                out_channel,
                out_channel,
                3,
                1,
                1,
            ),
        )

    def forward(self, x):
        return self.net(x) + self.skip(x)


class ResNetFullPreActivationBottleneckBlock(ResNetBlockBase):
    def __init__(self, in_channel, base_channel, expand_channel=None, expand_scale=4, **kwargs):
        expand_channel = expand_channel if expand_channel is not None else expand_scale*base_channel
        super().__init__(in_channel, expand_channel, **kwargs)

        self.net = nn.Sequential(
            self.Norm(in_channel),
            ReLU(),
            nn.Conv2d(
                in_channel,
                base_channel,
                1,
            ),
            self.Norm(base_channel),
            ReLU(),
            nn.Conv2d(
                base_channel,
                base_channel,
                3,
                self.stride,
                self.padding
            ),
            self.Norm(base_channel),
            ReLU(),
            nn.Conv2d(
                base_channel,
                expand_channel,
                1,
            ),
        )

    def forward(self, x):
        return self.net(x) + self.skip(x)


class ResNetBackBone(nn.Module):
    def __init__(self, config: Optional[ResNetConfig] = None) -> None:
        super().__init__()

        assert config is not None, f"[{self.__class__.__name__}] Please provide a configuration using `ResNetConfig`"

        self.base_block = self._get_base_block(config.base_block_name)

        self.conv_1 = nn.Sequential(
            nn.Conv2d(config.num_input_channel, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            ReLU()
        )
        self.conv_2 = nn.Sequential(OrderedDict(
            [('conv_2_maxpool', nn.MaxPool2d(3, stride=2, padding=1))] +
            list(self._build_conv_layer('conv_2', config.num_blocks_in_conv_layer['conv_2'], 64, 64, 256, False).items())
        ))
        self.conv_3 = self._build_conv_layer('conv_3', config.num_blocks_in_conv_layer['conv_3'], 256, 128, 512)
        self.conv_4 = self._build_conv_layer('conv_4', config.num_blocks_in_conv_layer['conv_4'], 512, 256, 1024)
        self.conv_5 = self._build_conv_layer('conv_5', config.num_blocks_in_conv_layer['conv_5'], 1024, 512, 2048)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        
        return self.conv_5(x)

    def _build_conv_layer(
        self, 
        layer_name: str, 
        num_base_blocks: int, 
        prev_channel: int, 
        base_channel: int, 
        expand_channel: int, 
        to_sequential: bool = True
    ) -> Union[nn.Sequential, OrderedDict]:

        conv_layer = OrderedDict([
            (
                f'{layer_name}_block_{idx}',
                self.base_block(
                    prev_channel if idx == 0 else expand_channel,
                    base_channel,
                    expand_channel,
                    stride=2 if idx == 0 and layer_name != 'conv_2' else 1,
                    padding=1,
                    remain_dim=True if idx == 0 and layer_name == 'conv_2' else False
                )
            ) for idx in range(num_base_blocks)
        ])

        return nn.Sequential(conv_layer) if to_sequential else conv_layer

    def _get_base_block(self, base_block_name: str = "") -> object:
        return globals()[base_block_name]

    
class ResNetWithLinearClassifier(nn.Module):
    def __init__(self, config: ResNetConfig = None) -> None:
        super().__init__()

        self.use_classifier = True if config.num_classes else False

        self.resnet_backbone = ResNetBackBone(config)

        if self.use_classifier:
            # Reference from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
            self.out_linear = nn.Sequential(OrderedDict([
                ("avg_pool", nn.AdaptiveAvgPool2d((1, 1))),
                ("flatten", Rearrange("b ... -> b (...)")),
                ("out_linear", nn.Linear(2048, config.num_classes))
            ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet_backbone(x)

        return self.out_linear(x)