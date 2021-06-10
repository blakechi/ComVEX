from functools import partial
from typing import Union, Optional
from collections import OrderedDict

import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange

from .config import BoTNetConfig
from comvex.utils import ResNetBlockBase, ResNetBlock, ResNetBottleneckBlock, ResNetFullPreActivationBlock, ResNetFullPreActivationBottleneckBlock, ResNetBackBone


ReLU = partial(nn.ReLU, inplace=True)


class BoTNetMHSA(nn.Module):
    def __init__(
        self, 
        *, 
        lateral_size, 
        dim, 
        heads, 
        head_dim=None, 
        **kwargs
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim if head_dim is not None else dim // heads

        assert (
            self.head_dim * self.heads == self.dim
        ), "Head dimension times the number of heads must be equal to embedding dimension"

        self.QKV = nn.Conv2d(dim, 3*dim, 1, bias=False)
        self.height_relative_pos = nn.Parameter(torch.empty(lateral_size, 1, dim))
        self.width_relative_pos = nn.Parameter(torch.empty(1, lateral_size, dim))
        nn.init.kaiming_uniform_(self.height_relative_pos)
        nn.init.kaiming_uniform_(self.width_relative_pos)

    def forward(self, x):
        b, c, h, w, p, device = *x.shape, self.heads, x.device

        q, k, v = self.QKV(x).chunk(chunks=3, dim=-3)
        q, k, v = map(lambda t: rearrange(t, "b (p d) h w -> b p (h w) d", p=p), (q, k, v))

        relative_pos = rearrange(self.height_relative_pos + self.width_relative_pos, "h w (p d) -> 1 p (h w) d", p=p)

        # Note: scaling doesn't be mentioned in the original paper
        qr = einsum("b p n d, l p m d -> b p n m", q, relative_pos)  # l means 1
        qk = einsum("b p n d, b p m d -> b p n m", q, k)

        similarity = (qr + qk).softmax(dim=-1)
        out = einsum("b p n m, b p m d -> b p n d", similarity, v)
        out = rearrange(out, "b p (h w) d -> b (p d) h w", h=h)

        return out


class BoTNetBlock(ResNetBlockBase):
    def __init__(
        self, 
        in_channel, 
        base_channel, 
        expand_channel, 
        *,
        lateral_size, 
        heads, 
        expand_scale=4, 
        **kwargs
    ):
        expand_channel = expand_channel if expand_channel is not None else expand_scale*base_channel
        super().__init__(in_channel, expand_channel, **kwargs)

        _layers = [
            nn.Conv2d(
                in_channel, 
                base_channel, 
                1
            ),
            self.Norm(num_features=base_channel),
            ReLU(),
            BoTNetMHSA(
                lateral_size=lateral_size,
                dim=base_channel,
                heads=heads, 
                **kwargs
            ),
            self.Norm(num_features=base_channel),
            ReLU(),
            nn.Conv2d(
                base_channel, 
                expand_channel, 
                1
            ),
            self.Norm(num_features=expand_channel),
        ]
        if 'stride' in kwargs and kwargs['stride'] == 2:  # If stride, add average pooling after BoTNetMHSA
            _layers.insert(
                *[(idx + 1) for idx, layer in enumerate(_layers) if isinstance(layer, BoTNetMHSA)], 
                nn.AvgPool2d(kernel_size=2, stride=2)
            )

        self.net = nn.Sequential(*_layers)
        self.relu = ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + self.skip(x))


class BoTNetFullPreActivationBlock(ResNetBlockBase):
    def __init__(
        self, 
        in_channel, 
        base_channel, 
        expand_channel, 
        *,
        lateral_size, 
        heads, 
        expand_scale=4, 
        **kwargs
    ):
        expand_channel = expand_channel if expand_channel is not None else expand_scale*base_channel
        super().__init__(in_channel, expand_channel, **kwargs)

        _layers = [
            self.Norm(num_features=in_channel),
            ReLU(),
            nn.Conv2d(
                in_channel, 
                base_channel, 
                1
            ),
            self.Norm(num_features=base_channel),
            ReLU(),
            BoTNetMHSA(
                lateral_size=lateral_size,
                dim=base_channel,
                heads=heads, 
                **kwargs
            ),
            self.Norm(num_features=base_channel),
            ReLU(),
            nn.Conv2d(
                base_channel, 
                expand_channel, 
                1
            ),
        ]
        if 'stride' in kwargs and kwargs['stride'] == 2:  # If stride, add average pooling after BoTNetMHSA
            _layers.insert(
                *[(idx + 1) for idx, layer in enumerate(_layers) if isinstance(layer, BoTNetMHSA)], 
                nn.AvgPool2d(kernel_size=2, stride=2)
            )

        self.net = nn.Sequential(*_layers)

    def forward(self, x):
        return self.net(x) + self.skip(x)


class BoTNetBackBone(nn.Module):
    def __init__(self, config: Optional[BoTNetConfig] = None) -> None:
        super().__init__()

        assert config is not None, f"[{self.__class__.__name__}] Please provide a configuration using `BoTNetConfig`"

        self.conv_base_block = self._get_base_block(config.conv_base_block_name)
        self.bot_base_block = self._get_base_block(config.bot_base_block_name)

        self.conv_1 = nn.Sequential(
            nn.Conv2d(config.num_input_channel, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            ReLU()
        )
        self.conv_2 = nn.Sequential(OrderedDict(
            [('conv_2_maxpool', nn.MaxPool2d(3, stride=2, padding=1))] +
            list(self._build_conv_layer('conv_2', config.num_blocks_in_layer['conv_2'], 64, 64, 256, False).items())
        ))
        self.conv_3 = self._build_conv_layer('conv_3', config.num_blocks_in_layer['conv_3'], 256, 128, 512)
        self.conv_4 = self._build_conv_layer('conv_4', config.num_blocks_in_layer['conv_4'], 512, 256, 1024)
        self.bot_layer = self._build_bot_layer(
            'bot', 
            config.num_blocks_in_layer['bot'], 
            1024, 
            512, 
            2048, 
            config.input_lateral_size // 2**5, 
            config.num_heads, 
            config.bot_block_indicator
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        
        return self.bot_layer(x)

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
                self.conv_base_block(
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

    def _build_bot_layer(
        self, 
        layer_name: str, 
        num_base_blocks: int, 
        prev_channel: int, 
        base_channel: int, 
        expand_channel: int,
        lateral_size: int,
        num_heads: int,
        bot_block_indicator: list,
        to_sequential: bool = True
    ) -> Union[nn.Sequential, OrderedDict]:

        bot_layer = OrderedDict([
            (
                f'{layer_name}_block_{idx}' if use_bot_block else f'conv_block_{idx}',
                self.bot_base_block(
                    prev_channel if idx == 0 else expand_channel,
                    base_channel,
                    expand_channel,
                    lateral_size=lateral_size*2 if idx == 0 else lateral_size,
                    heads=num_heads,
                    stride=2 if idx == 0 else 1,
                ) if use_bot_block else self.conv_base_block(
                    prev_channel if idx == 0 else expand_channel,
                    base_channel,
                    expand_channel,
                    stride=2 if idx == 0 else 1,
                    padding=1,
                )
            ) for idx, use_bot_block in enumerate(bot_block_indicator)
        ])

        return nn.Sequential(bot_layer) if to_sequential else bot_layer

    def _get_base_block(self, base_block_name: str = "") -> object:
        return globals()[base_block_name]


class BoTNetWithLinearClassifier(nn.Module):
    def __init__(self, config: BoTNetConfig = None) -> None:
        super().__init__()

        self.use_classifier = True if config.num_classes else False

        self.backbone = BoTNetBackBone(config)

        if self.use_classifier:
            # Reference from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
            self.out_linear = nn.Sequential(OrderedDict([
                ("avg_pool", nn.AdaptiveAvgPool2d((1, 1))),
                ("flatten", Rearrange("b ... -> b (...)")),
                ("out_linear", nn.Linear(2048, config.num_classes))
            ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)

        return self.out_linear(x)