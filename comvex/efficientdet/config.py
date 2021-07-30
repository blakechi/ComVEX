from typing import Tuple, List, Literal

from comvex.utils import ConfigBase, EfficientNetBackboneConfig, BiFPNConfig


class EfficientDetBackboneConfig(ConfigBase):
    def __init__(
        self,
        efficientnet_backbone_config: EfficientNetBackboneConfig,
        bifpn_num_layers: int,
        bifpn_channel: int,
        image_shapes: List[Tuple[int]],
        dimension: int = 2,
        upsample_mode: Literal["nearest", "linear", "bilinear", "bicubic", "trilinear"] = "nearest",
        use_bias: bool = False,
        use_batch_norm: bool = False,
        norm_mode: Literal["fast_norm", "softmax", "channel_fast_norm", "channel_softmax"] = "fast_norm",
        batch_norm_epsilon: float = 1e-5,
        batch_norm_momentum: float = 1e-1,
    ) -> None:
        super().__init__()

        self.efficientnet_backbone_config = efficientnet_backbone_config

        bifpn_config = BiFPNConfig(
            bifpn_num_layers=bifpn_num_layers,
            bifpn_channel=bifpn_channel,
            channels_in_stages=[],
            image_shapes=image_shapes,
            dimension=dimension,
            upsample_mode=upsample_mode,
            use_bias=use_bias,
            use_batch_norm=use_batch_norm,
            norm_mode=norm_mode,
            batch_norm_epsilon=batch_norm_epsilon,
            batch_norm_momentum=batch_norm_momentum,
        )
        for name, value in bifpn_config.__dict__.items():
            if name != "channels_in_stages":  # `channels_in_stages` will be handled automatically in `EfficientDetBackbone`
                setattr(self, name, value)

    @classmethod
    def D0(cls, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.B0(resolution=512),
            3,
            64,
            image_shapes=(512, 512),
            **kwargs
        )
        
    @classmethod
    def D1(cls, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.B1(resolution=640),
            4,
            88,
            image_shapes=(640, 640),
            **kwargs
        )

    @classmethod
    def D2(cls, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.B2(resolution=768),
            5,
            112,
            image_shapes=(768, 768),
            **kwargs
        )

    @classmethod
    def D3(cls, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.B3(resolution=896),
            6,
            120,
            image_shapes=(896, 896),
            **kwargs
        )

    @classmethod
    def D4(cls, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.B4(resolution=1024),
            7,
            224,
            image_shapes=(1024, 1024),
            **kwargs
        )

    @classmethod
    def D5(cls, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.B5(resolution=1280),
            7,
            288,
            image_shapes=(1280, 1280),
            **kwargs
        )

    @classmethod
    def D6(cls, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.B6(resolution=1280),
            8,
            384,
            image_shapes=(1280, 1280),
            **kwargs
        )

    @classmethod
    def D7(cls, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.B6(resolution=1536),
            8,
            384,
            image_shapes=(1536, 1536),
            **kwargs
        )

    @classmethod
    def D7x(cls, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.B7(resolution=1536),
            8,
            384,
            image_shapes=(1536, 1536),
            **kwargs
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
    def D0(cls, num_classes: int, num_anchors: int, **kwargs) -> "EfficientDetObjectDetectionConfig":
        
        return cls(
            EfficientDetBackboneConfig.D0(),
            num_pred_layers=3,
            num_classes=num_classes,
            num_anchors=num_anchors,
            **kwargs
        )
        
    @classmethod
    def D1(cls, num_classes: int, num_anchors: int, **kwargs) -> "EfficientDetObjectDetectionConfig":
        return cls(
            EfficientDetBackboneConfig.D1(),
            num_pred_layers=3,
            num_classes=num_classes,
            num_anchors=num_anchors,
            **kwargs
        )

    @classmethod
    def D2(cls, num_classes: int, num_anchors: int, **kwargs) -> "EfficientDetObjectDetectionConfig":
        return cls(
            EfficientDetBackboneConfig.D2(),
            num_pred_layers=3,
            num_classes=num_classes,
            num_anchors=num_anchors,
            **kwargs
        )

    @classmethod
    def D3(cls, num_classes: int, num_anchors: int, **kwargs) -> "EfficientDetObjectDetectionConfig":
        return cls(
            EfficientDetBackboneConfig.D3(),
            num_pred_layers=4,
            num_classes=num_classes,
            num_anchors=num_anchors,
            **kwargs
        )

    @classmethod
    def D4(cls, num_classes: int, num_anchors: int, **kwargs) -> "EfficientDetObjectDetectionConfig":
        return cls(
            EfficientDetBackboneConfig.D4(),
            num_pred_layers=4,
            num_classes=num_classes,
            num_anchors=num_anchors,
            **kwargs
        )

    @classmethod
    def D5(cls, num_classes: int, num_anchors: int, **kwargs) -> "EfficientDetObjectDetectionConfig":
        return cls(
            EfficientDetBackboneConfig.D5(),
            num_pred_layers=4,
            num_classes=num_classes,
            num_anchors=num_anchors,
            **kwargs
        )

    @classmethod
    def D6(cls, num_classes: int, num_anchors: int, **kwargs) -> "EfficientDetObjectDetectionConfig":
        return cls(
            EfficientDetBackboneConfig.D6(),
            num_pred_layers=5,
            num_classes=num_classes,
            num_anchors=num_anchors,
            **kwargs
        )

    @classmethod
    def D7(cls, num_classes: int, num_anchors: int, **kwargs) -> "EfficientDetObjectDetectionConfig":
        return cls(
            EfficientDetBackboneConfig.D7(),
            num_pred_layers=5,
            num_classes=num_classes,
            num_anchors=num_anchors,
            **kwargs
        )

    @classmethod
    def D7x(cls, num_classes: int, num_anchors: int, **kwargs) -> "EfficientDetObjectDetectionConfig":
        return cls(
            EfficientDetBackboneConfig.D7x(),
            num_pred_layers=5,
            num_classes=num_classes,
            num_anchors=num_anchors,
            **kwargs
        )