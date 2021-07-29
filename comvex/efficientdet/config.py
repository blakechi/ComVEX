from typing import Tuple, List, Literal

from comvex.utils import ConfigBase, EfficientNetBackboneConfig, BiFPNConfig


class EfficientDetBackboneConfig(ConfigBase):
    def __init__(
        self,
        efficientnet_config: EfficientNetBackboneConfig,
        bifpn_num_layers: int,
        bifpn_channel: int,
        channels_in_stages: List[int],
        image_shapes: List[Tuple[int]],
        dimension: int = 2,
        upsample_mode: Literal["nearest", "linear", "bilinear", "bicubic", "trilinear"] = "nearest",
        use_bias: bool = False,
        use_batch_norm: bool = False,
        norm_mode: Literal["fast_norm", "softmax", "channel_fast_norm", "channel_softmax"] = "fast_norm",
    ) -> None:
        super().__init__()

        self.efficientnet_config = efficientnet_config

        bifpn_config = BiFPNConfig(
            bifpn_num_layers=bifpn_num_layers,
            bifpn_channel=bifpn_channel,
            channels_in_stages=channels_in_stages,
            image_shapes=image_shapes,
            dimension=dimension,
            upsample_mode=upsample_mode,
            use_bias=use_bias,
            use_batch_norm=use_batch_norm,
            norm_mode=norm_mode,
        )
        for name, value in bifpn_config.__dict__.items():
            setattr(self, name, value)

    @classmethod
    def D0(cls, channels_in_stages, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.B0(resolution=512),
            3, 64, channels_in_stages, image_shape=(512, 512), **kwargs
        )
        
    @classmethod
    def D1(cls, channels_in_stages, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.B1(resolution=640),
            4, 88, channels_in_stages, image_shape=(640, 640), **kwargs
        )

    @classmethod
    def D2(cls, channels_in_stages, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.B2(resolution=768),
            5, 112, channels_in_stages, image_shape=(768, 768), **kwargs
        )

    @classmethod
    def D3(cls, channels_in_stages, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.B3(resolution=896),
            6, 120, channels_in_stages, image_shape=(896, 896), **kwargs
        )

    @classmethod
    def D4(cls, channels_in_stages, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.B4(resolution=1024),
            7, 224, channels_in_stages, image_shape=(1024, 1024), **kwargs
        )

    @classmethod
    def D5(cls, channels_in_stages, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.B5(resolution=1280),
            7, 288, channels_in_stages, image_shape=(1280, 1280), **kwargs
        )

    @classmethod
    def D6(cls, channels_in_stages, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.B6(resolution=1280),
            8, 384, channels_in_stages, image_shape=(1280, 1280), **kwargs
        )

    @classmethod
    def D7(cls, channels_in_stages, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.B6(resolution=1536),
            8, 384, channels_in_stages, image_shape=(1536, 1536), **kwargs
        )

    @classmethod
    def D7x(cls, channels_in_stages, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.B7(resolution=1536),
            8, 384, channels_in_stages, image_shape=(1536, 1536), **kwargs
        )

        
class EfficientDetObjectDetectionConfig(ConfigBase):
    def __init__(
        self,
        efficientdet_backbone_config: EfficientDetBackboneConfig,
        num_pred_layers: int,
        num_classes: int,
        num_anchors: int,
        use_seperable_conv: bool = True,
        path_dropout: float = 0.,
    ) -> None:
        super().__init__()

        self.efficientdet_backbone_config = efficientdet_backbone_config
        self.num_pred_layers = num_pred_layers
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.use_seperable_conv = use_seperable_conv
        self.path_dropout = path_dropout

    @classmethod
    def D0(cls, channels_in_stages: List[int], num_classes: int, num_anchors: int, **backbone_kwargs) -> "EfficientDetObjectDetectionConfig":
        return cls(
            EfficientDetBackboneConfig.D0(channels_in_stages, **backbone_kwargs),
            num_pred_layers=3,
            num_classes=num_classes,
            num_anchors=num_anchors,
        )
        
    @classmethod
    def D1(cls, channels_in_stages: List[int], num_classes: int, num_anchors: int, **backbone_kwargs) -> "EfficientDetObjectDetectionConfig":
        return cls(
            EfficientDetBackboneConfig.D1(channels_in_stages, **backbone_kwargs),
            num_pred_layers=3,
            num_classes=num_classes,
            num_anchors=num_anchors,
        )

    @classmethod
    def D2(cls, channels_in_stages: List[int], num_classes: int, num_anchors: int, **backbone_kwargs) -> "EfficientDetObjectDetectionConfig":
        return cls(
            EfficientDetBackboneConfig.D2(channels_in_stages, **backbone_kwargs),
            num_pred_layers=3,
            num_classes=num_classes,
            num_anchors=num_anchors,
        )

    @classmethod
    def D3(cls, channels_in_stages: List[int], num_classes: int, num_anchors: int, **backbone_kwargs) -> "EfficientDetObjectDetectionConfig":
        return cls(
            EfficientDetBackboneConfig.D3(channels_in_stages, **backbone_kwargs),
            num_pred_layers=4,
            num_classes=num_classes,
            num_anchors=num_anchors,
        )

    @classmethod
    def D4(cls, channels_in_stages: List[int], num_classes: int, num_anchors: int, **backbone_kwargs) -> "EfficientDetObjectDetectionConfig":
        return cls(
            EfficientDetBackboneConfig.D4(channels_in_stages, **backbone_kwargs),
            num_pred_layers=4,
            num_classes=num_classes,
            num_anchors=num_anchors,
        )

    @classmethod
    def D5(cls, channels_in_stages: List[int], num_classes: int, num_anchors: int, **backbone_kwargs) -> "EfficientDetObjectDetectionConfig":
        return cls(
            EfficientDetBackboneConfig.D5(channels_in_stages, **backbone_kwargs),
            num_pred_layers=4,
            num_classes=num_classes,
            num_anchors=num_anchors,
        )

    @classmethod
    def D6(cls, channels_in_stages: List[int], num_classes: int, num_anchors: int, **backbone_kwargs) -> "EfficientDetObjectDetectionConfig":
        return cls(
            EfficientDetBackboneConfig.D6(channels_in_stages, **backbone_kwargs),
            num_pred_layers=5,
            num_classes=num_classes,
            num_anchors=num_anchors,
        )

    @classmethod
    def D7(cls, channels_in_stages: List[int], num_classes: int, num_anchors: int, **backbone_kwargs) -> "EfficientDetObjectDetectionConfig":
        return cls(
            EfficientDetBackboneConfig.D7(channels_in_stages, **backbone_kwargs),
            num_pred_layers=5,
            num_classes=num_classes,
            num_anchors=num_anchors,
        )

    @classmethod
    def D7x(cls, channels_in_stages: List[int], num_classes: int, num_anchors: int, **backbone_kwargs) -> "EfficientDetObjectDetectionConfig":
        return cls(
            EfficientDetBackboneConfig.D7x(channels_in_stages, **backbone_kwargs),
            num_pred_layers=5,
            num_classes=num_classes,
            num_anchors=num_anchors,
        )