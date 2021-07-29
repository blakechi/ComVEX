from comvex.utils.helpers.functions import name_with_msg
from collections import OrderedDict
from typing import Literal, List, Tuple, Optional

import torch
from torch import nn
from torch.nn import functional as F
try:
    from typing_extensions import Final
except:
    from torch.jit import Final

from comvex.utils import SeperableConvXd, XXXConvXdBase, ConfigBase
from comvex.utils.helpers import get_conv_layer


@torch.jit.script
def bifpn_fast_norm(x, weights, dim=0):
    weights = F.relu(weights)
    norm = weights.sum(dim, keepdim=True)

    return x*weights / (norm + 1e-4)


@torch.jit.script
def bifpn_softmax(x, weights, dim=0):
    weights = F.softmax(weights, dim)

    return (x*weights).sum(dim=0)


class BiFPNConfig(ConfigBase):
    def __init__(
        self,
        num_layers: int,
        bifpn_channel: int,
        channels_in_stages: List[int],
        shapes_in_stages: Optional[List[Tuple[int]]] = None,
        image_shape: Optional[Tuple[int]] = None,
        shape_scales: List[int] = [8, 16, 32, 64, 128],
        dimension: int = 2,
        upsample_mode: Literal["nearest", "linear", "bilinear", "bicubic", "trilinear"] = "nearest",
        use_bias: bool = False,
        use_batch_norm: bool = False,
        norm_mode: Literal["fast_norm", "softmax", "channel_fast_norm", "channel_softmax"] = "fast_norm",
        batch_norm_epsilon: float = 1e-5,
        batch_norm_momentum: float = 1e-1,
    ) -> None:
        super().__init__()

        assert (
            shapes_in_stages is not None or image_shape is not None
        ), name_with_msg("Either `shapes_in_shapes` or `image_shape` should be specified")

        if image_shape is not None:
            shapes_in_stages = [(image_shape[0] // scale, image_shape[1] // scale) for scale in shape_scales]

        self.num_layers = num_layers
        self.bifpn_channel = bifpn_channel
        self.channels_in_stages = channels_in_stages
        self.shapes_in_stages = shapes_in_stages 
        self.dimension = dimension
        self.upsample_mode = upsample_mode
        self.use_bias = use_bias
        self.use_batch_norm = use_batch_norm
        self.norm_mode = norm_mode
        self.batch_norm_epsilon = batch_norm_epsilon
        self.batch_norm_momentum = batch_norm_momentum

    @classmethod
    def BiFPN_Default(cls, num_layers: int, bifpn_channel: int, channels_in_stages: List[int], image_shape: Tuple[int], **kwargs) -> "BiFPNConfig":
        return cls(
            num_layers,
            bifpn_channel,
            channels_in_stages,
            image_shape=image_shape
        )
        

class BiFPNResizeXd(XXXConvXdBase):
    r"""The `Resize` in equations of Section 3.3 in the official paper.
    Reference from: https://github.com/google/automl/blob/0fb012a80487f0defa4446957c8faf878cd9b75b/efficientdet/efficientdet_arch.py#L55-L95.

    Support 1, 2, or 3D inputs.
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        in_shape: Tuple[int],
        out_shape: Tuple[int],
        dimension: int = 2,
        upsample_mode: Literal["nearest", "linear", "bilinear", "bicubic", "trilinear"] = "nearest",
        use_conv_after_downsampling: bool = True,
        use_bias: bool = False,
        batch_norm_epsilon: float = 1e-5,
        batch_norm_momentum: float = 1e-1,
    ) -> None:

        assert (
            len(in_shape) == len(out_shape)
        ), name_with_msg(f"The length of input shape mush be qual to the output one. But got: `in_shape` = {in_shape} and `out_shape` = {out_shape}")

        assert (
            (in_shape[0] > out_shape[0]) == (in_shape[1] > out_shape[1])
        ), name_with_msg(f"`Elements in `in_shape` must be all larger or small than `out_shape`. But got: `in_shape` = {in_shape} and `out_shape` = {out_shape}")

        extra_components = { "max_pool": "AdaptiveMaxPool" } 
        extra_components = { **extra_components, "batch_norm": "BatchNorm" } if use_conv_after_downsampling else extra_components
        super().__init__(in_channel, out_channel, dimension, extra_components=extra_components)
        
        if in_shape[0] > out_shape[0]:  # downsampling
            self.interpolate_shape = self.max_pool(out_shape)
            self.downsampling = True
            self.use_conv_after_downsampling = use_conv_after_downsampling
            if self.use_conv_after_downsampling:
                self.proj_channel = self.conv(in_channel, out_channel, kernel_size=1, use_bias=use_bias)
                self.norm = self.batch_norm(
                    out_channel,
                    eps=batch_norm_epsilon,
                    momentum=batch_norm_momentum,
                )
        else:  # upsampling
            self.downsampling = False
            self.interpolate_shape = nn.Upsample(out_shape, mode=upsample_mode, align_corners=True)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.interpolate_shape(x)

        if self.downsampling and self.use_conv_after_downsampling:
            x = self.proj_channel(x)
            x = self.norm(x)

        return x


class BiFPNNodeBase(nn.Module):
    r"""
    Referennce from: https://github.com/google/automl/blob/0fb012a80487f0defa4446957c8faf878cd9b75b/efficientdet/efficientdet_arch.py#L418-L475
    """
    def __init__(
        self,
        num_inputs: int,
        in_channel: int,
        out_channel: int,
        in_shape: Tuple[int],
        out_shape: Tuple[int],
        dimension: int = 2,
        upsample_mode: Literal["nearest", "linear", "bilinear", "bicubic", "trilinear"] = "nearest",
        use_bias: bool = False,
        use_batch_norm: bool = False,
        norm_mode: Literal["fast_norm", "softmax", "channel_fast_norm", "channel_softmax"] = "fast_norm",
        **possible_batch_norm_kwargs
    ) -> None:
        super().__init__()

        # Project features' channels only at the intermediae nodes of the first layer in BiFPN to `bifpn_channel`
        self.proj_feature_channel = get_conv_layer(f"Conv{dimension}d")(
            in_channel,
            out_channel,
            kernel_size=1,
        ) if in_channel != out_channel else nn.Identity()

        self.resize = BiFPNResizeXd(
            in_channel,
            out_channel,
            in_shape,
            out_shape,
            dimension,
            upsample_mode,
            use_bias,
            use_batch_norm,
            **possible_batch_norm_kwargs
        )
        self.conv = SeperableConvXd(
            in_channel,
            out_channel,
            dimension=dimension,
            **possible_batch_norm_kwargs
        )

        if norm_mode.endswith("fast_norm"):
            self.fuse_features = bifpn_fast_norm
        elif norm_mode.endswith("softmax"):
            self.fuse_features = bifpn_softmax
        else:
            raise ValueError(name_with_msg(f"Unknown `norm_mode`. Got: `norm_mode` = {norm_mode}"))

        if norm_mode.startswith("channel"):
            self.weights = nn.Parameter(torch.ones(num_inputs, 1, out_channel, *([1]*len(in_shape))))  # Ex: (2, 1, C, 1, 1) for images with shape: (B, C, H, W)
        else:
            self.weights = nn.Parameter(torch.ones(num_inputs))


class BiFPNIntermediateNode(BiFPNNodeBase):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(num_inputs=2, **kwargs)

    def forward(self, x: torch.Tensor, x_diff: torch.Tensor) -> torch.Tensor:
        x = self.proj_feature_channel(x)
        x_diff = self.resize(x_diff)
        x_stack = torch.stack([x, x_diff], dim=0)

        return self.conv(self.fuse_features(x_stack, self.weights))


BiFPNOutputEndPoint = BiFPNIntermediateNode  #The start and end nodes in the outputs


class BiFPNOutputNode(BiFPNNodeBase):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(num_inputs=3, **kwargs)

    def forward(self, x: torch.Tensor, x_hidden: torch.Tensor, x_diff: torch.Tensor) -> torch.Tensor:
        x = self.proj_feature_channel(x)
        x_diff = self.resize(x_diff)
        x_stack = torch.stack([x, x_hidden, x_diff], dim=0)

        return self.conv(self.fuse_features(x_stack, self.weights))


class BiFPNLayer(nn.Module):
    r"""BiFPNLayer
    The same in Figure. 3 of the official paper.
    """

    num_nodes: Final[int]

    def __init__(
        self,
        bifpn_channel: int,
        shapes_in_stages: List[Tuple[int]],
        channels_in_stages: Optional[List[int]] = None,
        **possible_batch_norm_kwargs,
    ) -> None:
        super().__init__()

        self.num_nodes = len(shapes_in_stages)

        self.intermediate_nodes = nn.ModuleList(OrderedDict([
            (
                f"intermediate_node_{idx}",
                BiFPNIntermediateNode(
                    in_channel=channels_in_stages[idx + 1] if channels_in_stages is not None else bifpn_channel,  # Channel of the feature map comes from deeper layers.
                    out_channel=bifpn_channel,
                    in_shape=shapes_in_stages[idx + 1],
                    out_shape=shapes_in_stages[idx],
                    **possible_batch_norm_kwargs
                )
            ) for idx in range(1, self.num_nodes - 1)
        ]))
        self.output_nodes = nn.ModuleList(OrderedDict([
            (
                f"output_node_{idx}",
                BiFPNOutputEndPoint(
                    in_channel=bifpn_channel,
                    out_channel=bifpn_channel,
                    in_shape=shapes_in_stages[idx + 1] if idx ==0 else shapes_in_stages[idx - 1],
                    out_shape=shapes_in_stages[idx],
                    **possible_batch_norm_kwargs
                ) if idx == 0 or idx == self.num_nodes else BiFPNOutputNode(
                    in_channel=bifpn_channel,
                    out_channel=bifpn_channel,
                    in_shape=shapes_in_stages[idx - 1],
                    out_shape=shapes_in_stages[idx],
                    **possible_batch_norm_kwargs
                )
            ) for idx in range(self.num_nodes)
        ]))

    def forward(self, feature_list: List[torch.Tensor]) -> List[torch.Tensor]:
        hidden_feature_list = []
        out_feature_list = []
        x_diff = feature_list[-1]  # It will be propagated to every nodes

        # Intermediate Nodes
        for rev_idx, node in reversed(enumerate(self.intermediate_nodes)):
            x = feature_list[rev_idx + 1]  # plus 1 because intermediate nodes don't include shallowest and deepest features
            x_diff = node(x, x_diff)
            hidden_feature_list.append(x_diff)

        # Output Nodes
        for idx, node in enumerate(self.output_nodes):
            x = feature_list[idx]
            if idx == 0 or idx == self.num_nodes:
                x_diff = node(x, x_diff)
            else:
                x_hidden = hidden_feature_list[-(idx + 1)]  # select reversely
                x_diff = node(x, x_hidden, x_diff)

            out_feature_list.append(x_diff)

        return out_feature_list
            

class BiFPN(nn.Module):
    r"""BiFPN from EfficientDet (https://arxiv.org/abs/1911.09070)

    Note: The `feature_list` is assumed to be ordered from shallow to deep features.
    """
    def __init__(
        self,
        num_layers: int,
        bifpn_channel: int,
        channels_in_stages: List[int],
        shapes_in_stages: List[Tuple[int]],
        dimension: int = 2,
        upsample_mode: Literal["nearest", "linear", "bilinear", "bicubic", "trilinear"] = "nearest",
        use_bias: bool = False,
        use_batch_norm: bool = False,
        norm_mode: Literal["fast_norm", "softmax", "channel_fast_norm", "channel_softmax"] = "fast_norm",
        batch_norm_epsilon: float = 1e-5,
        batch_norm_momentum: float = 1e-1
    ) -> None:
        super().__init__()

        assert (
            len(shapes_in_stages) == len(channels_in_stages)
        ), name_with_msg(f"The length of `shapes_in_stages` and `channels_in_stages` must be equal. But got: {len(shapes_in_stages)} for shapes and {len(channels_in_stages)} for channels.")

        self.layers = nn.ModuleList(OrderedDict([
            (
                f"layer_{idx}",
                BiFPNLayer(
                    bifpn_channel,
                    shapes_in_stages,
                    channels_in_stages if idx == 0 else None,
                    dimension,
                    upsample_mode,
                    use_bias,
                    use_batch_norm,
                    norm_mode,
                    batch_norm_epsilon=batch_norm_epsilon,
                    batch_norm_momentum=batch_norm_momentum
                )
            ) for idx in range(num_layers)
        ]))

    def forward(self, feature_list: List[torch.Tensor]) -> List[torch.Tensor]:
        for layer in self.layers:
            feature_list = layer(feature_list)

        return feature_list