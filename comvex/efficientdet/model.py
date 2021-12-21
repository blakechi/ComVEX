from functools import partial
from collections import OrderedDict
from itertools import accumulate
from typing import List, Tuple, Optional
from typing_extensions import Literal

import torch
from torch import nn
try:
    from typing_extensions import Final
except:
    from torch.jit import Final

from comvex.utils import EfficientNetBackboneConfig, EfficientNetBackbone, BiFPN, SeperableConvXd, PathDropout
from comvex.utils.helpers import get_norm_layer, get_act_fnc, config_pop_argument, name_with_msg
from .config import EfficientDetObjectDetectionConfig
 

_DEFAULT_ACT_FNC = "SiLU"


class EfficientDetBackbone(nn.Module):
    r"""
    `EfficientNetBackbone` + `BiFPN`
    """

    _DEFAULT_FEATURE_MAP_SCALE: Final[List[int]] = [8, 16, 32, 64, 128]  # Official paper: Figure. 3 - EfficientNet Backbone
    
    def __init__(
        self,
        efficientnet_backbone_config: EfficientNetBackboneConfig,
        image_shapes: Tuple[int],
        bifpn_num_layers: int,
        bifpn_channel: int,
        dimension: int = 2,
        upsample_mode: Literal["nearest", "linear", "bilinear", "bicubic", "trilinear"] = "nearest",
        use_bias: bool = False,
        use_conv_after_downsampling: bool = False,
        norm_mode: Literal["fast_norm", "softmax", "channel_fast_norm", "channel_softmax"] = "fast_norm",
        batch_norm_epsilon: float = 1e-5,
        batch_norm_momentum: float = 1e-1,
        feature_map_indices: Optional[List[int]] = None
    ) -> None:
        config_pop_argument(efficientnet_backbone_config, "return_feature_maps")
        super().__init__()

        self.efficentnet_backbone = EfficientNetBackbone(
            return_feature_maps=True,
            **efficientnet_backbone_config.__dict__
        )

        channels_in_stages = self.efficentnet_backbone.channels
        strides_in_stages = self.efficentnet_backbone.strides
        scale_in_stages = list(accumulate(strides_in_stages, lambda x, y: x*y))
        shapes_in_stages = [(image_shapes[0] // scale, image_shapes[1] // scale) for scale in scale_in_stages]

        num_stages = len(channels_in_stages)
        unique_scale_with_idx = {}
        for idx, scale in enumerate(scale_in_stages):
            unique_scale_with_idx[scale] = idx

        unique_scale_with_idx = list(unique_scale_with_idx.items())
        feature_map_indices = feature_map_indices or [idx for scale, idx in unique_scale_with_idx if scale in self._DEFAULT_FEATURE_MAP_SCALE]
        assert (
            max(feature_map_indices) < num_stages
        ), name_with_msg(f"Indices in `feature_map_indices` ({feature_map_indices}) are larger than the number of existing feature maps.")

        self.feature_map_keys = [f"stage_{idx + 1}" for idx in feature_map_indices]
        channels_in_nodes = [channels_in_stages[idx] for idx in feature_map_indices]
        shapes_in_nodes = [shapes_in_stages[idx] for idx in feature_map_indices]

        self.bifpn = BiFPN(
            bifpn_num_layers,
            bifpn_channel,
            channels_in_nodes,
            shapes_in_nodes,
            dimension,
            upsample_mode,
            use_bias,
            use_conv_after_downsampling,
            norm_mode,
            batch_norm_epsilon,
            batch_norm_momentum,
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feature_map_dict = self.efficentnet_backbone(x)

        feature_maps = [feature_map_dict[key] for key in self.feature_map_keys]
        feature_maps = self.bifpn(feature_maps)

        return feature_maps


class EfficientDetPredictionHead(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_feature_maps: int,
        in_channel: int,
        out_channel: int,
        dimension: int = 2,
        act_fnc_name: str = _DEFAULT_ACT_FNC,
        use_seperable_conv: bool = True,
        path_dropout: float = 0.,
    ) -> None:
        super().__init__()

        self.conv = partial(SeperableConvXd, kernel_size=3, padding=1, use_conv_only=True) if use_seperable_conv else partial(nn.Conv2d, kernel_size=3, padding=1)
        self.norm = get_norm_layer(f"BatchNorm{dimension}d")
        self.act_fnc = partial(get_act_fnc(act_fnc_name), inplace=True)()

        self.conv_layers = nn.ModuleList([self.conv(in_channel, in_channel) for _ in range(num_layers)])
        self.feature_map_norms = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(OrderedDict([
                    (f"batch_norm_{feature_map_idx}_{layer_idx}", self.norm(in_channel)),
                ])) for layer_idx in range(num_layers)
            ]) for feature_map_idx in range(num_feature_maps)
        ])

        self.out_conv = self.conv(in_channel, out_channel)
        self.path_dropout = PathDropout(path_dropout)

    def forward(self, feature_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        outs = []
        for idx, feature_map_norm in enumerate(self.feature_map_norms):  # Feature maps have their own batch norm layers
            x = feature_maps[idx]

            for layer_idx, (conv, norm) in enumerate(zip(self.conv_layers, feature_map_norm)):  # but share conv layers
                x_prev = x
                x = conv(x)
                x = norm(x)
                x = self.act_fnc(x)
            
                if layer_idx > 0:  # From https://github.com/google/automl/blob/master/efficientdet/efficientdet_arch.py#L180-L182
                    x = x_prev + self.path_dropout(x)

            x = self.out_conv(x)  # and share the output layer as well
            outs.append(x)

        return outs


class EfficientDetClassNet(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_classes: int,
        num_anchors: int,
        in_channel: int,
        num_feature_maps: int = 5,
        dimension: int = 2,
        act_fnc_name: str = _DEFAULT_ACT_FNC,
        use_seperable_conv: bool = True,
        path_dropout: float = 0.,
    ) -> None:
        super().__init__()
    
        self.head = EfficientDetPredictionHead(
            num_layers,
            num_feature_maps,
            in_channel,
            num_classes*num_anchors,
            dimension,
            act_fnc_name,
            use_seperable_conv,
            path_dropout,
        )

    def forward(self, feature_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        return self.head(feature_maps)


class EfficientDetBoxNet(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_anchors: int,
        in_channel: int,
        num_feature_maps: int = 5,
        dimension: int = 2,
        act_fnc_name: str = _DEFAULT_ACT_FNC,
        use_seperable_conv: bool = True,
        path_dropout: float = 0.,
    ) -> None:
        super().__init__()
    
        self.head = EfficientDetPredictionHead(
            num_layers,
            num_feature_maps,
            in_channel,
            4*num_anchors,
            dimension,
            act_fnc_name,
            use_seperable_conv,
            path_dropout,
        )

    def forward(self, feature_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        return self.head(feature_maps)


class EfficientDetObjectDetection(nn.Module):
    r"""
    """
    
    def __init__(self, config: EfficientDetObjectDetectionConfig) -> None:
        num_pred_layers = config_pop_argument(config, "num_pred_layers")
        num_classes = config_pop_argument(config, "num_classes")
        num_anchors = config_pop_argument(config, "num_anchors")
        use_seperable_conv = config_pop_argument(config, "use_seperable_conv")
        path_dropout = config_pop_argument(config, "path_dropout")
        super().__init__()

        self.backbone = EfficientDetBackbone(**config.efficientdet_backbone_config.__dict__)

        num_feature_maps = 5 or len(config.efficientdet_backbone_config.feature_map_indices)  # 5 is the default
        self.class_net = EfficientDetClassNet(
            num_layers=num_pred_layers,
            num_classes=num_classes,
            num_anchors=num_anchors,
            in_channel=config.efficientdet_backbone_config.bifpn_channel,
            num_feature_maps=num_feature_maps,
            use_seperable_conv=use_seperable_conv,
            path_dropout=path_dropout,
        )
        self.box_net = EfficientDetBoxNet(
            num_layers=num_pred_layers,
            num_anchors=num_anchors,
            in_channel=config.efficientdet_backbone_config.bifpn_channel,
            num_feature_maps=num_feature_maps,
            use_seperable_conv=use_seperable_conv,
            path_dropout=path_dropout,
        )

    def forward(self, x):
        x = self.backbone(x)

        return self.class_net(x), self.box_net(x)


class EfficientDetForSemanticSegmentation(nn.Module):
    r"""
    """
    
    def __init__(
        self,
    ) -> None:
        super().__init__()
