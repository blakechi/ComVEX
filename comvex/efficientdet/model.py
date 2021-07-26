from comvex.utils.dropout import PathDropout
from comvex.utils.helpers.functions import get_act_fnc, get_norm_layer, name_with_msg
from functools import partial
from collections import OrderedDict, namedtuple
from os import name
from typing import Literal, Optional, Union, List, Tuple, Dict,

import torch
from torch import nn
from torch.nn import functional as F
try:
    from typing_extensions import Final
except:
    from torch.jit import Final

from comvex.utils import EfficientNetBackbone, BiFPN, SeperableConvXd
from comvex.utils.helpers import get_norm_layer, get_act_fnc get_attr_if_exists, config_pop_argument
from .config import EfficientDetConfig
 

class EfficientDetPredictionHead(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_feature_maps: int,
        in_channel: int,
        out_channel: int,
        dimension: int = 2,
        act_fnc_name: str = "ReLU",
        use_seperable_conv: bool = True,
        path_dropout: float = 0.,
    ) -> None:
        super().__init__()

        self.conv = partial(SeperableConvXd, kernel_size=3, padding=1, use_conv_only=True) if use_seperable_conv else partial(nn.Conv2d, kernel_size=3, padding=1)
        self.norm = get_norm_layer(f"BatchNorm{dimension}d")
        self.act_fnc = partial(get_act_fnc(act_fnc_name), inplace=True)

        self.conv_layers = nn.ModuleList([self.conv(in_channel, in_channel) for _ in range(num_layers)])
        self.feature_map_norms_act_fnc = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(OrderedDict([
                    (f"batch_norm_{feature_map_idx}_{layer_idx}", self.norm(in_channel)),
                    (f"act_fnc_{feature_map_idx}_{layer_idx}", self.act_fnc())
                ])) for layer_idx in range(num_layers)
            ]) for feature_map_idx in range(num_feature_maps)
        ])

        self.out_conv = self.conv(in_channel, out_channel)
        self.path_dropout = PathDropout(path_dropout)

    def forward(self, feature_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        outs = []
        for idx, norm_act_fncs in self.feature_map_norms_act_fnc:  # Feature maps have their own batch norm layers
            x = feature_maps[idx]

            for layer_idx, conv, norm_act_fnc in enumerate(zip(self.conv_layers, norm_act_fncs)):  # but share conv layers
                x_prev = x
                x = conv(x)
                x = norm_act_fnc(x)
            
                if layer_idx > 0:  # From https://github.com/google/automl/blob/master/efficientdet/efficientdet_arch.py#L180-L182
                    x = x_prev + self.path_dropout(x)

            x = self.out_conv(x)  # and share the output layer as well
            outs.append(x)

        return outs


class EfficientDetClassNet(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_feature_maps: int,
        feature_map_channel: int,
        num_classes: int,
        num_anchors: int,
        dimension: int = 2,
        act_fnc_name: str = "ReLU",
        use_seperable_conv: bool = True,
        path_dropout: float = 0.,
    ) -> None:
        super().__init__()
    
        self.head = EfficientDetPredictionHead(
            num_layers,
            num_feature_maps,
            feature_map_channel,
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
        num_feature_maps: int,
        feature_map_channel: int,
        num_anchors: int,
        dimension: int = 2,
        act_fnc_name: str = "ReLU",
        use_seperable_conv: bool = True,
        path_dropout: float = 0.,
    ) -> None:
        super().__init__()
    
        self.head = EfficientDetPredictionHead(
            num_layers,
            num_feature_maps,
            feature_map_channel,
            4*num_anchors,
            dimension,
            act_fnc_name,
            use_seperable_conv,
            path_dropout,
        )

    def forward(self, feature_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        return self.head(feature_maps)


class EfficientDetBackbone(nn.Module):
    r"""
    `EfficientNetBackbone` + `BiFPN`
    """
    
    def __init__(
        self,
        backbone: nn.Module,
    ) -> None:
        super().__init__()


class EfficientDetForObjectDetection(nn.Module):
    r"""
    
    """
    
    def __init__(
        self,
    ) -> None:
        super().__init__()


class EfficientDetForSemanticSegmentation(nn.Module):
    r"""

    """
    
    def __init__(
        self,
    ) -> None:
        super().__init__()