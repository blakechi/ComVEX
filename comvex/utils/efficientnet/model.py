from collections import OrderedDict
from typing import Optional, Union, List, Tuple, Dict
import math

import torch
from torch import nn

from comvex.utils import PathDropout
from comvex.utils.helpers import name_with_msg, get_attr_if_exists


class EfficientNetBase(nn.Module):
    __constants__ = [
        "layers",
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
        up_sampling_mode: Optional[str] = None,  # Check out: https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html?highlight=up%20sample#torch.nn.Upsample
        return_feature_maps: bool = False,
        se_scale: float = 0.25,
    ) -> None:
        super().__init__()

        # Table 1. from the official paper (all stages)
        self.num_layers = self.scale_and_round_layers([1, 1, 2, 2, 3, 3, 4, 1, 1], depth_scale)
        self.channels = self.scale_and_round_channels([32, 16, 24, 40, 80, 112, 192, 320, 1280], width_scale)
        self.kernel_sizes = [3, 3, 3, 5, 3, 5, 5, 3, 1]    
        self.strides = [1, 1, 2, 2, 2, 2, 1, 2, 1]
        self.expand_scales = [None, 1, 6, 6, 6, 6, 6, 6, None]
        self.se_scales = [None, *(se_scale)*7, None]

        self.resolution = resolution
        # From: https://github.com/tensorflow/tpu/blob/3679ca6b979349dde6da7156be2528428b000c7c/models/official/efficientnet/preprocessing.py#L88
        # The default for resizing is `bicubic`
        self.up_sampling = nn.Upsample(self.resolution, mode=up_sampling_mode) if up_sampling_mode else None
        self.return_feature_maps = return_feature_maps

    def scale_and_round_layers(self, in_list: List[int], scale) -> List[int]:
        out = map(lambda x: int(math.ceil(x*scale)), in_list)
        out[0] = 1  # Stage 1 always has one layer

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

        filters = map(lambda x: x*scale, in_list)

        new_filters = map(lambda x: max(min_depth, int(x + depth_divisor / 2) // depth_divisor * depth_divisor), filters)

        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += depth_divisor
        new_filters = map(lambda new_x, x: int(new_x + depth_divisor) if new_x < 0.9*x else int(new_x), (new_filters, filters))
        
        return new_filters


class SeperateConvXd(nn.Module):
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
    ):
        super().__init__()

        assert (
            (0 < dimension) and (dimension < 4)
        ), name_with_msg(self, "`dimension` must be larger than 0 and smaller than 4")

        if dimension == 1:
            conv = nn.Conv1d
        elif dimension == 2:
            conv = nn.Conv2d
        else: 
            conv = nn.Conv3d

        self.depth_wise_conv = nn.Sequential(
            conv(
                in_channel, 
                in_channel*kernels_per_layer, 
                kernel_size, 
                padding=padding, 
                groups=in_channel
            ),
            get_attr_if_exists(nn, norm_layer_name)(
                in_channel*kernels_per_layer,
                **kwargs
            ),
            get_attr_if_exists(nn, act_fnc_name)()
        )
        self.pixel_wise_conv = nn.Sequential(
            conv(
                in_channel*kernels_per_layer,
                out_channel,
                kernel_size=1,
            ),
            get_attr_if_exists(nn, norm_layer_name)(
                in_channel*kernels_per_layer,
                **kwargs
            )
        )

    def forward(self, x):
        x = self.depth_wise_conv(x)
        x = self.pixel_wise_conv(x)

        return x


class MBConvXd(nn.Module):
    r"""
    Reference from: 
        1. MobileNetV2 (https://arxiv.org/pdf/1801.04381.pdf)
        2. MnasNet (https://arxiv.org/pdf/1807.11626.pdf)
        3. Squeeze-and-Excitation Networks (https://arxiv.org/pdf/1709.01507.pdf)
        4. Official Implementation (https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py)
    
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
        path_dropout: float = 0.,
        se_scale: Optional[float] = None,
        se_act_fnc_name: str = "SiLU",
        dimension: int = 2,
        **kwargs  # For example: `eps` and `elementwise_affine` for `nn.LayerNorm`
    ):
        super().__init__()

        out_channel = out_channel if out_channel is not None else in_channel

        assert (
            (0 < dimension) and (dimension < 4)
        ), name_with_msg(self, "`dimension` must be larger than 0 and smaller than 4")

        if dimension == 1:
            conv = nn.Conv1d
        elif dimension == 2:
            conv = nn.Conv2d
        else: 
            conv = nn.Conv3d

        assert (
            expand_channel is not None or expand_scale is not None
        ), name_with_msg(self, "Either `expand_channel` or `expand_scale` should be specified")
        expand_channel = expand_channel if expand_channel is not None else in_channel*expand_scale

        #
        self.pixel_wise_conv_0 = nn.Sequential(
            conv(
                in_channel,
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

        #
        self.depth_wise_conv = nn.Sequential(
            conv(
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

        #
        self.se_block = None
        if se_scale:
            bottleneck_channel = int(expand_channel*se_scale)

            self.se_block = nn.Sequential(
                nn.AdaptiveMaxPool2d((1, 1)),
                nn.Conv2d(expand_channel, bottleneck_channel, 1),
                get_attr_if_exists(nn, se_act_fnc_name)(),
                nn.Conv2d(bottleneck_channel, expand_channel, 1),
                nn.Sigmoid(),
            )

        #
        self.pixel_wise_conv_1 = nn.Sequential(
            conv(
                expand_channel,
                out_channel,
                kernel_size=1,
                bias=False,
            ),
            get_attr_if_exists(nn, norm_layer_name)(
                out_channel,
                **kwargs
            )
        )

        #
        self.path_dropout = PathDropout(path_dropout)
            
    def forward(self, x):

        # First pixel-wise conv
        res = self.pixel_wise_conv_0(x)
        
        # Depth-wise conv
        res = self.depth_wise_conv(res)

        # SE block
        if self.se_block is not None:
            res = res*self.se_block(res)

        # Second pixel-wise conv
        res = self.pixel_wise_conv_1(res)
            
        return x + self.path_dropout(res)

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
        path_dropout: float = 0.,
        # From: https://github.com/tensorflow/tpu/blob/3679ca6b979349dde6da7156be2528428b000c7c/models/official/efficientnet/efficientnet_builder.py#L187-L188
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.99,
        return_feature_map: bool = False,
    ) -> None:
        super().__init__(depth_scale, width_scale, resolution, up_sampling_mode, return_feature_map)

        kwargs = {}
        kwargs["path_dropout"] = path_dropout
        kwargs["batch_norm_eps"] = batch_norm_eps
        kwargs["batch_norm_momentum"] = batch_norm_momentum
        kwargs["act_fnc_name"] = act_fnc_name
        kwargs["se_act_fnc_name"] = se_act_fnc_name
        
        self.stage_1 = self._build_stage("1", kwargs,
            in_channel=image_channel,
            out_channel=self.channels[0],
            kernel_size=3
        )
        self.stage_2 = self._build_stage("2", kwargs)
        self.stage_3 = self._build_stage("3", kwargs)
        self.stage_4 = self._build_stage("4", kwargs)
        self.stage_5 = self._build_stage("5", kwargs)
        self.stage_6 = self._build_stage("6", kwargs)
        self.stage_7 = self._build_stage("7", kwargs)
        self.stage_8 = self._build_stage("8", kwargs)
        self.stage_9 = self._build_stage("9", kwargs,
            in_channel=self.channels[-2],
            out_channel=self.channels[-1],
            kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> Union[
        Tuple[torch.Tensor, Dict[str, torch.Tensor]],
        torch.Tensor
    ]:
        # These `if` statements should be removed after scripting, so don't worry
        if self.up_sampling:
            x = self.up_sampling(x)

        if self.return_feature_maps:
            endpoints = {}

        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        if self.return_feature_maps:
            endpoints['reduction_1'] = x

        x = self.stage_4(x)
        if self.return_feature_maps:
            endpoints['reduction_2'] = x

        x = self.stage_5(x)
        if self.return_feature_maps:
            endpoints['reduction_3'] = x

        x = self.stage_6(x)
        x = self.stage_7(x)
        if self.return_feature_maps:
            endpoints['reduction_4'] = x

        x = self.stage_8(x)
        x = self.stage_9(x)
        if self.return_feature_maps:
            endpoints['reduction_5'] = x
            
        return (x, endpoints) if self.return_feature_maps else x

    def _build_stage(self, stage_idx: str, **kwargs) -> nn.Module:
        access_idx = int(stage_idx) - 1  # Since the naming of stages is not 0-based

        if 0 < access_idx and access_idx < 8:  # If it's Stage 2 ~ 8
            return nn.Sequential(OrderedDict([
                (
                    f"stage_{stage_idx}_layer_{idx}",
                    MBConvXd(
                        in_channel=self.channels[access_idx - 1],
                        out_channel=self.channels[access_idx],
                        expand_scale=self.expand_scales[access_idx],
                        kernel_size=self.kernel_sizes[access_idx],
                        stride=self.strides[access_idx] if idx == 0 else 1,  # only for the first MBConvXd block
                        padding=self.strides[access_idx] // 2 if idx == 0 else 0,  # only for the first MBConvXd block, (stride = 1 -> padding = 0), (stride = 3 -> padding = 1), (stride = 5 -> padding = 2)
                        se_scale=self.se_scales[access_idx],
                        **kwargs
                    )
                ) for idx in range(self.num_layers[access_idx])
            ]))

        else:
            return nn.Sequential(OrderedDict([
                (
                    f"stage_{stage_idx}_layer_0",
                    nn.Sequential(
                        nn.Conv2d(kwargs['in_channel'], kwargs['out_channel'], kwargs['kernel_size'], bias=False),
                        nn.BatchNorm2d(
                            kwargs['out_channel'],
                            eps=kwargs['batch_norm_eps'],
                            momentum=kwargs['batch_norm_momentum'],
                        ),
                        get_attr_if_exists(nn.Module, kwargs['act_fnc_name'])()
                    )
                )
            ]))