from functools import partial
from collections import OrderedDict
from typing import Optional, Union, Tuple, Dict

import torch
from torch import nn

from comvex.utils import EfficientNetBase, MBConvXd
from comvex.utils.helpers import get_attr_if_exists, config_pop_argument
from .config import EfficientNetV2Config, EfficientNetV2BaseConfig


FusedMBConvXd = partial(MBConvXd, expansion_head_type="fused")


class EfficientNetV2Base(EfficientNetBase):
    def __init__(
        self,
        base_config: EfficientNetV2BaseConfig,
        depth_scale: float,
        width_scale: float,
        train_resolution: int,
        eval_resolution: int,
        se_scale: float = 0.25,
        up_sampling_mode: Optional[str] = None,  # Not recommmand, should be done by `torchvision.transforms`
        return_feature_maps: bool = False,
    ) -> None:
        # What if we don't init EfficientBEtBase but nn.Module only?
        super().__init__(
            depth_scale,
            width_scale,
            train_resolution,
            se_scale=se_scale,
            up_sampling_mode=up_sampling_mode,
            return_feature_maps=return_feature_maps,
        )

        # Table 4. from the official paper (all stages)
        # set: num_layers, channels, kernel_sizes, strides, expand_scales, se_scales
        for key, value in base_config.__dict__.items():
            setattr(self, key, value)
            
            if key == "num_layers":
                self.num_layers = self.scale_and_round_layers(self.num_layers, depth_scale)

            if key == "channels":
                self.channels = self.scale_and_round_channels(self.channels, width_scale)
            

        self.__dict__.pop("resolution")  # remove `resolution` for `EfficientNet`
        self.train_resolution = train_resolution
        self.eval_resolution = eval_resolution


class EfficientNetV2Backbone(EfficientNetV2Base):
    def __init__(
        self,
        base_config: EfficientNetV2BaseConfig,
        image_channel: int,
        depth_scale: float,
        width_scale: float,
        train_resolution: int,
        eval_resolution: int,
        up_sampling_mode: Optional[str] = None,
        act_fnc_name: str = "SiLU",
        se_act_fnc_name: str = "SiLU",
        se_scale: float = 0.25,
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.99,
        return_feature_maps: bool = False,
        path_dropout: float = 0.2,  # From: https://github.com/google/automl/blob/master/efficientnetv2/hparams.py#L234, and it doesn't be overrided later.
    ) -> None:
        super().__init__(
            base_config,
            depth_scale,
            width_scale,
            train_resolution,
            eval_resolution,
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

        num_stages = len(self.num_layers)
        self.stages = nn.ModuleList([
            self._build_stage(  # The first and last stages
                stage_idx,
                **kwargs,
                in_channel=image_channel if stage_idx == 0 else self.channels[-2],
                out_channel=self.channels[0 if stage_idx == 0 else -1],
                kernel_size=self.kernel_sizes[0 if stage_idx == 0 else -1]
            ) if stage_idx == 0 or stage_idx == (num_stages - 1) else self._build_stage(  # Stages in between
                stage_idx,
                **kwargs
            )
            for stage_idx in range(num_stages)
        ])
            
    def forward(self, x: torch.Tensor) -> Union[
        Tuple[torch.Tensor, Dict[str, torch.Tensor]],
        torch.Tensor
    ]:
        # These `if` statements should be removed after scripting, so don't worry
        if self.up_sampling_mode:
            x = nn.functional.interpolate(
                x,
                size=self.train_resolution if self.training else self.eval_resolution,
                mode=self.up_sampling_mode,
            )

        if self.return_feature_maps:
            feature_maps = {}

        for stage_idx, stage in enumerate(self.stages):
            x = stage(x)
            
            if self.return_feature_maps and self.strides[stage_idx] == 2:  # If be asked to return the feature map and H and W shrink
                feature_maps[f"stage_{stage_idx}"] = x

        return (x, feature_maps) if self.return_feature_maps else x

    def _build_stage(self, stage_idx: int, **kwargs) -> nn.Module:
        num_stages = len(self.num_layers)

        if 0 < stage_idx and stage_idx < (num_stages - 1):
            path_dropout = kwargs.pop("path_dropout")
            conv_block = FusedMBConvXd if self.se_scales[stage_idx] is None else MBConvXd

            return nn.Sequential(OrderedDict([
                (
                    f"stage_{stage_idx}_layer_{idx}",
                    conv_block(
                        in_channel=self.channels[stage_idx - 1] if idx == 0 else self.channels[stage_idx],
                        out_channel=self.channels[stage_idx],
                        expand_scale=self.expand_scales[stage_idx],
                        kernel_size=self.kernel_sizes[stage_idx],
                        # only for the first MBConvXd block
                        stride=self.strides[stage_idx] if idx == 0 else 1,
                        padding=self.kernel_sizes[stage_idx] // 2,
                        se_scale=self.se_scales[stage_idx],
                        # Inverted `survival_prob` from: https://github.com/tensorflow/tpu/blob/3679ca6b979349dde6da7156be2528428b000c7c/models/official/efficientnet/efficientnet_model.py#L659
                        path_dropout=path_dropout*(stage_idx + 1) / (num_stages - 2),  # 2 for the first and last stage
                        **kwargs
                    ) 
                ) for idx in range(self.num_layers[stage_idx])
            ]))

        else:
            return nn.Sequential(OrderedDict([
                (
                    f"stage_{stage_idx}_layer_0",
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


class EfficientNetV2WithLinearClassifier(EfficientNetV2Backbone):
    r"""
    Same as `EfficientNetV2WithLinearClassifier` with different parent and config classes
    """
    def __init__(self, config: EfficientNetV2Config = None) -> None:
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