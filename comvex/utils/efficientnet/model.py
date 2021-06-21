from typing import Optional, Union, List, Tuple, Dict, Literal
from collections import OrderedDict
import math

import torch
from torch import nn

from comvex.utils import PathDropout, XXXConvXdBase
from comvex.utils.helpers import name_with_msg, get_attr_if_exists, config_pop_argument
from .config import EfficientNetConfig


class EfficientNetBase(nn.Module):
    __constants__ = [
        "num_layers",
        "channels",
        "kernel_sizes",
        "strides",
        "expand_scales",
        "se_scales",
        "resolution",
        "return_feature_maps",
    ]

    def __init__(
        self,
        depth_scale: float,
        width_scale: float,
        resolution: int,
        se_scale: float = 0.25,
        up_sampling_mode: Optional[str] = None,  # Check out: https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html?highlight=up%20sample#torch.nn.Upsample
        return_feature_maps: bool = False,
    ) -> None:
        super().__init__()

        # Table 1. from the official paper (all stages)
        self.num_layers = self.scale_and_round_layers([1, 1, 2, 2, 3, 3, 4, 1, 1], depth_scale)
        self.channels = self.scale_and_round_channels([32, 16, 24, 40, 80, 112, 192, 320, 1280], width_scale)
        self.kernel_sizes = [3, 3, 3, 5, 3, 5, 5, 3, 1]    
        self.strides = [1, 2, 1, 2, 2, 2, 1, 2, 1]
        self.expand_scales = [None, 1, 6, 6, 6, 6, 6, 6, None]
        self.se_scales = [None, *((se_scale,)*7), None]

        self.resolution = resolution
        # From: https://github.com/tensorflow/tpu/blob/3679ca6b979349dde6da7156be2528428b000c7c/models/official/efficientnet/preprocessing.py#L88
        # The default for resizing is `bicubic`
        self.up_sampling_mode = up_sampling_mode
        self.return_feature_maps = return_feature_maps

    def scale_and_round_layers(self, in_list: List[int], scale) -> List[int]:
        out = list(map(lambda x: int(math.ceil(x*scale)), in_list))
        out[0] = 1  # Stage 1 always has one layer
        out[-1] = 1  # Stage 9 always has one layer
        
        return out

    def scale_and_round_channels(self, in_list: List[int], scale) -> List[int]:
        r"""
        Reference from: https://github.com/tensorflow/tpu/blob/3679ca6b979349dde6da7156be2528428b000c7c/models/official/efficientnet/efficientnet_model.py#L106
        According to: 
            1. https://github.com/tensorflow/tpu/blob/3679ca6b979349dde6da7156be2528428b000c7c/models/official/efficientnet/condconv/efficientnet_condconv_builder.py#L70-L71
            2. https://github.com/tensorflow/tpu/blob/3679ca6b979349dde6da7156be2528428b000c7c/models/official/efficientnet/efficientnet_builder.py#L195-L196
            3. https://github.com/tensorflow/tpu/blob/3679ca6b979349dde6da7156be2528428b000c7c/models/official/efficientnet/main.py
        Since:
            1. `depth_divisor` and `min_depth` are constant values with 8 and None 
            2. `min_depth = min_depth or divisor`
            3. They don't be overrided in the training setup 
        => set them both to 8

        Not sure what's the purpose of `depth_divisor` and `min_depth`...
        """
        depth_divisor, min_depth = 8, 8

        filters = list(map(lambda x: x*scale, in_list))
        new_filters = list(map(lambda x: max(min_depth, int(x + depth_divisor / 2) // depth_divisor * depth_divisor), filters))

        # Make sure that round down does not go down by more than 10%.
        new_filters = list(map(lambda xs: int(xs[0] + depth_divisor) if xs[0] < 0.9*xs[1] else int(xs[0]), zip(new_filters, filters)))

        return new_filters


class SeperateConvXd(XXXConvXdBase):
    r"""
    Reference from: MnasNet (https://arxiv.org/pdf/1807.11626.pdf)
    """

    def __init__(
        self, 
        in_channel: int, 
        out_channel: int, 
        kernel_size: int = 3, 
        padding: int = 1, 
        kernels_per_layer: int = 1, 
        norm_layer_name: str = "BatchNorm2d",
        act_fnc_name: str = "ReLU",
        dimension: int = 2,
        **kwargs  # For the normalization layer
    ) -> None:
        super().__init__(in_channel, out_channel, dimension=dimension)

        self.depth_wise_conv = nn.Sequential(
            self.conv(
                self.in_channel, 
                self.in_channel*kernels_per_layer, 
                kernel_size, 
                padding=padding, 
                groups=self.in_channel
            ),
            get_attr_if_exists(nn, norm_layer_name)(
                self.in_channel*kernels_per_layer,
                **kwargs
            ),
            get_attr_if_exists(nn, act_fnc_name)()
        )
        self.pixel_wise_conv = nn.Sequential(
            self.conv(
                self.in_channel*kernels_per_layer,
                out_channel,
                kernel_size=1,
            ),
            get_attr_if_exists(nn, norm_layer_name)(
                self.in_channel*kernels_per_layer,
                **kwargs
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_wise_conv(x)
        x = self.pixel_wise_conv(x)

        return x


class SEConvXd(XXXConvXdBase):
    r"""
    Squeeze-and-Excitation Convolution 

    From: Squeeze-and-Excitation Networks (https://arxiv.org/pdf/1709.01507.pdf)
    """
    def __init__(
        self,
        in_channel: int,
        bottleneck_channel: int,
        out_channel: Optional[int] = None,
        se_act_fnc_name: Optional[str] = None,
        dimension: int = 2,
    ) -> None:
        super().__init__(in_channel, out_channel, dimension=dimension)

        self.layers = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            self.conv(self.in_channel, bottleneck_channel, 1),
            get_attr_if_exists(nn, se_act_fnc_name)(),
            self.conv(bottleneck_channel, self.out_channel, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x*self.layers(x)        


class MBConvXd(XXXConvXdBase):
    r"""
    Reference from: 
        1. MobileNetV2 (https://arxiv.org/pdf/1801.04381.pdf)
        2. MnasNet (https://arxiv.org/pdf/1807.11626.pdf)
        3. Official Implementation (https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py)
    
    Note: `Swish` is called `SiLU` in PyTorch
    """

    def __init__(
        self, 
        in_channel: int,
        out_channel: Optional[int] = None,
        expand_channel: Optional[int] = None,
        expand_scale: Optional[int] = None, 
        kernel_size: int = 3, 
        stride: int = 1,
        padding: int = 1, 
        norm_layer_name: str = "BatchNorm2d",
        act_fnc_name: str = "SiLU",
        se_scale: Optional[float] = None,
        se_act_fnc_name: str = "SiLU",
        dimension: int = 2,
        path_dropout: float = 0.,
        expansion_head_type: Literal["pixel_depth", "fused"] = "pixel_depth",
        **kwargs  # For example: `eps` and `elementwise_affine` for `nn.LayerNorm`
    ):
        super().__init__(in_channel, out_channel, dimension=dimension)

        assert (
            expand_channel is not None or expand_scale is not None
        ), name_with_msg(self, "Either `expand_channel` or `expand_scale` should be specified")
        expand_channel = expand_channel if expand_channel is not None else in_channel*expand_scale

        assert (
            isinstance(expansion_head_type, str) and expansion_head_type in ["pixel_depth", "fused"]
        ), name_with_msg(
            f"The specified `expansion_head_type` - {expansion_head_type} ({type(expansion_head_type)}) doesn't exist.\n \
            Please choose from here: ['pixel_depth', 'fused']"
        )

        # Expansion Head
        if expansion_head_type == "pixel_depth":
            pixel_wise_conv_0 = nn.Sequential(
                self.conv(
                    self.in_channel,
                    expand_channel,
                    kernel_size=1,
                    bias=False
                ),
                get_attr_if_exists(nn, norm_layer_name)(
                    expand_channel,
                    **kwargs
                ),
                get_attr_if_exists(nn, act_fnc_name)()
            )

            depth_wise_conv = nn.Sequential(
                self.conv(
                    expand_channel, 
                    expand_channel, 
                    kernel_size, 
                    stride=stride,
                    padding=padding, 
                    groups=expand_channel,
                    bias=False
                ),
                get_attr_if_exists(nn, norm_layer_name)(
                    expand_channel,
                    **kwargs
                ),
                get_attr_if_exists(nn, act_fnc_name)()
            )

            self.expansion_head = nn.Sequential(
                pixel_wise_conv_0,
                depth_wise_conv
            )
        else:
            self.expansion_head = nn.Sequential(
                nn.Conv2d(
                    self.in_channel,
                    expand_channel,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False
                ),
                get_attr_if_exists(nn, norm_layer_name)(
                    expand_channel,
                    **kwargs
                ),
                get_attr_if_exists(nn, act_fnc_name)()
            )

        #
        self.se_block = None
        if se_scale is not None:
            bottleneck_channel = int(expand_channel*se_scale)

            self.se_block = SEConvXd(
                expand_channel,
                bottleneck_channel,
                se_act_fnc_name=se_act_fnc_name,
            )

        #
        self.pixel_wise_conv_1 = nn.Sequential(
            self.conv(
                expand_channel,
                self.out_channel,
                kernel_size=1,
                bias=False,
            ),
            get_attr_if_exists(nn, norm_layer_name)(
                self.out_channel,
                **kwargs
            )
        )

        # From: https://github.com/tensorflow/tpu/blob/3679ca6b979349dde6da7156be2528428b000c7c/models/official/efficientnet/utils.py#L276
        # It's a batch-wise dropout
        self.path_dropout = PathDropout(path_dropout)
        self.skip = True if self.in_channel == self.out_channel and stride == 1 else False

    def forward(self, x):

        # expansion head
        res = self.expansion_head(x)

        # SE block
        if self.se_block is not None:
            res = self.se_block(res)

        # Second pixel-wise conv
        res = self.pixel_wise_conv_1(res)
        
        # Path Dropout
        res = self.path_dropout(res)

        return x + res if self.skip else res

    @classmethod
    def MBConv2d6k3(cls, in_channel: int, out_channel: int, **kwargs) -> "MBConvXd":
        r"""
        The MBConv6 (k3x3) from MnasNet (https://arxiv.org/pdf/1807.11626.pdf)
        """
        return cls(
            in_channel,
            out_channel,
            expand_scale=6,
            kernel_size=3,
            expansion_head_type="pixel_depth",
            **kwargs
        )


class EfficientNetBackbone(EfficientNetBase):
    def __init__(
        self,
        image_channel: int,
        depth_scale: float,
        width_scale: float,
        resolution: int,
        up_sampling_mode: Optional[str] = None,
        act_fnc_name: str = "SiLU",
        se_act_fnc_name: str = "SiLU",
        se_scale: float = 0.25,
        # From: https://github.com/tensorflow/tpu/blob/3679ca6b979349dde6da7156be2528428b000c7c/models/official/efficientnet/efficientnet_builder.py#L187-L188
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.99,
        return_feature_maps: bool = False,
        # Can be overrided here in `EfficientNet`: https://github.com/tensorflow/tpu/blob/3679ca6b979349dde6da7156be2528428b000c7c/models/official/efficientnet/main.py#L256
        # But from `EfficientNetV2`: https://github.com/google/automl/blob/master/efficientnetv2/hparams.py#L234, it's 0.2 and doesn't be overrided later.
        path_dropout: float = 0.2,
    ) -> None:
        super().__init__(
            depth_scale,
            width_scale,
            resolution,
            se_scale=se_scale,
            up_sampling_mode=up_sampling_mode,
            return_feature_maps=return_feature_maps,
        )

        kwargs = {}
        kwargs["path_dropout"] = path_dropout
        kwargs["eps"] = batch_norm_eps
        kwargs["momentum"] = batch_norm_momentum
        kwargs["act_fnc_name"] = act_fnc_name
        kwargs["se_act_fnc_name"] = se_act_fnc_name
        
        self.stage_1 = self._build_stage("1", **kwargs,
            in_channel=image_channel,
            out_channel=self.channels[0],
            kernel_size=self.kernel_sizes[0]
        )
        self.stage_2 = self._build_stage("2", **kwargs)
        self.stage_3 = self._build_stage("3", **kwargs)
        self.stage_4 = self._build_stage("4", **kwargs)
        self.stage_5 = self._build_stage("5", **kwargs)
        self.stage_6 = self._build_stage("6", **kwargs)
        self.stage_7 = self._build_stage("7", **kwargs)
        self.stage_8 = self._build_stage("8", **kwargs)
        self.stage_9 = self._build_stage("9", **kwargs,
            in_channel=self.channels[-2],
            out_channel=self.channels[-1],
            kernel_size=self.kernel_sizes[-1]
        )

    def forward(self, x: torch.Tensor) -> Union[
        Tuple[torch.Tensor, Dict[str, torch.Tensor]],
        torch.Tensor
    ]:
        # These `if` statements should be removed after scripting, so don't worry
        if self.up_sampling_mode:
            x = nn.functional.interpolate(
                x,
                size=self.resolution,
                mode=self.up_sampling_mode
            )

        if self.return_feature_maps:
            feature_maps = {}

        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        if self.return_feature_maps:
            feature_maps['reduction_1'] = x

        x = self.stage_4(x)
        if self.return_feature_maps:
            feature_maps['reduction_2'] = x

        x = self.stage_5(x)
        if self.return_feature_maps:
            feature_maps['reduction_3'] = x

        x = self.stage_6(x)
        x = self.stage_7(x)
        if self.return_feature_maps:
            feature_maps['reduction_4'] = x

        x = self.stage_8(x)
        x = self.stage_9(x)
        if self.return_feature_maps:
            feature_maps['reduction_5'] = x
            
        return (x, feature_maps) if self.return_feature_maps else x

    def _build_stage(self, stage_idx: str, **kwargs) -> nn.Module:
        access_idx = int(stage_idx) - 1  # Since the naming of stages is not 0-based
        num_stages = len(self.num_layers)
        
        if 0 < access_idx and access_idx < 8:  # If it's Stage 2 ~ 8
            path_dropout = kwargs.pop("path_dropout")

            return nn.Sequential(OrderedDict([
                (
                    f"layer_{idx}",
                    MBConvXd(
                        in_channel=self.channels[access_idx - 1] if idx == 0 else self.channels[access_idx],
                        out_channel=self.channels[access_idx],
                        expand_scale=self.expand_scales[access_idx],
                        kernel_size=self.kernel_sizes[access_idx],
                        # only for the first MBConvXd block
                        stride=self.strides[access_idx] if idx == 0 else 1,
                        # (kernel_size = 1 -> padding = 0), (kernel_size = 3 -> padding = 1), (kernel_size = 5 -> padding = 2)
                        padding=self.kernel_sizes[access_idx] // 2,
                        se_scale=self.se_scales[access_idx],
                        # Inverted `survival_prob` from: https://github.com/tensorflow/tpu/blob/3679ca6b979349dde6da7156be2528428b000c7c/models/official/efficientnet/efficientnet_model.py#L659
                        path_dropout=path_dropout*(access_idx + 1) / (num_stages - 2),  # 2 for the first and last stage
                        **kwargs
                    )
                ) for idx in range(self.num_layers[access_idx])
            ]))

        else:
            return nn.Sequential(OrderedDict([
                (
                    f"layer_0",
                    nn.Sequential(
                        nn.Conv2d(
                            kwargs['in_channel'],
                            kwargs['out_channel'],
                            kwargs['kernel_size'],
                            padding=kwargs['kernel_size'] // 2,
                            bias=False
                        ),
                        nn.BatchNorm2d(
                            kwargs['out_channel'],
                            eps=kwargs['eps'],
                            momentum=kwargs['momentum'],
                        ),
                        get_attr_if_exists(nn, kwargs['act_fnc_name'])()
                    )
                )
            ]))

    def num_parameters(self) -> int:
        return sum(params.numel() for _, params in self.named_parameters())


class EfficientNetWithLinearClassifier(EfficientNetBackbone):
    def __init__(self, config: EfficientNetConfig = None) -> None:
        ff_dropout = config_pop_argument(config, "ff_dropout")
        num_classes = config_pop_argument(config, "num_classes")
        super().__init__(**config.__dict__)

        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        self.ff_dropout = nn.Dropout(ff_dropout)
        self.proj_head = nn.Linear(
            self.channels[-1],
            num_classes
        )

    def forward(self, x: torch.Tensor) -> Union[
        Tuple[torch.Tensor, Dict[str, torch.Tensor]],
        torch.Tensor
    ]:
        b, *rest = x.shape
        if self.return_feature_maps:
            x, feature_maps = super().forward(x)
        else:
            x = super().forward(x)

        # (B, C, H, W) -> (B, C, 1, 1) -> (B, C), avoid einops here now for scripting, edit it when einops is scriptable
        x = self.pooler(x).view(b, -1)

        x = self.ff_dropout(x)
        x = self.proj_head(x)

        return (x, feature_maps) if self.return_feature_maps else x

