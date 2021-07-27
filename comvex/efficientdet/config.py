from typing import Tuple, List, Literal

from comvex.utils import ConfigBase, EfficientNetBackboneConfig, BiFPNConfig


class EfficientDetBackboneConfig(ConfigBase):
    def __init__(
        self,
        efficientnet_config: EfficientNetBackboneConfig,
        bifpn_config: BiFPNConfig
    ) -> None:
        super().__init__()

        self.efficientnet_config = efficientnet_config
        self.bifpn_config = BiFPNConfig

    @classmethod
    def D0(cls, channels_in_stages, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.D0(resolution=512),
            BiFPNConfig.BiFPN_Default(3, 64, channels_in_stages, image_shape=(512, 512), **kwargs)
        )
        
    @classmethod
    def D1(cls, channels_in_stages, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.D1(resolution=640),
            BiFPNConfig.BiFPN_Default(4, 88, channels_in_stages, image_shape=(640, 640), **kwargs)
        )

    @classmethod
    def D2(cls, channels_in_stages, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.D2(resolution=768),
            BiFPNConfig.BiFPN_Default(5, 112, channels_in_stages, image_shape=(768, 768), **kwargs)
        )

    @classmethod
    def D3(cls, channels_in_stages, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.D3(resolution=896),
            BiFPNConfig.BiFPN_Default(6, 120, channels_in_stages, image_shape=(896, 896), **kwargs)
        )

    @classmethod
    def D4(cls, channels_in_stages, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.D4(resolution=1024),
            BiFPNConfig.BiFPN_Default(7, 224, channels_in_stages, image_shape=(1024, 1024), **kwargs)
        )

    @classmethod
    def D5(cls, channels_in_stages, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.D5(resolution=1280),
            BiFPNConfig.BiFPN_Default(7, 288, channels_in_stages, image_shape=(1280, 1280), **kwargs)
        )

    @classmethod
    def D6(cls, channels_in_stages, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.D6(resolution=1280),
            BiFPNConfig.BiFPN_Default(8, 384, channels_in_stages, image_shape=(1280, 1280), **kwargs)
        )

    @classmethod
    def D7(cls, channels_in_stages, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.D6(resolution=1536),
            BiFPNConfig.BiFPN_Default(8, 384, channels_in_stages, image_shape=(1536, 1536), **kwargs)
        )

    @classmethod
    def D7x(cls, channels_in_stages, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.D7(resolution=1536),
            BiFPNConfig.BiFPN_Default(8, 384, channels_in_stages, image_shape=(1536, 1536), **kwargs)
        )

        
class EfficientDetObjectDetectionConfig(ConfigBase):
    def __init__(
        self,
        efficientdet_backbone_config: EfficientDetBackboneConfig,
        num_pred_layers: int,
        num_classes: int,
        num_anchors: int,
        path_dropout: float = 0.,
    ) -> None:
        super().__init__()

        self.efficientdet_backbone_config = efficientdet_backbone_config
        self.num_pred_layers = num_pred_layers
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.path_dropout = path_dropout

    @classmethod
    def D0(cls, channels_in_stages: List[int], num_classes: int, num_anchors: int, **backbone_kwargs) -> "EfficientDetObjectDetectionConfig":
        return cls(
            EfficientNetBackboneConfig.B0(channels_in_stages, **backbone_kwargs),
            num_pred_layers=3,
            num_classes=num_classes,
            num_anchors=num_anchors,
        )
        
    @classmethod
    def D1(cls, channels_in_stages: List[int], num_classes: int, num_anchors: int, **backbone_kwargs) -> "EfficientDetObjectDetectionConfig":
        return cls(
            EfficientNetBackboneConfig.B1(channels_in_stages, **backbone_kwargs),
            num_pred_layers=3,
            num_classes=num_classes,
            num_anchors=num_anchors,
        )

    @classmethod
    def D2(cls, channels_in_stages: List[int], num_classes: int, num_anchors: int, **backbone_kwargs) -> "EfficientDetObjectDetectionConfig":
        return cls(
            EfficientNetBackboneConfig.B2(channels_in_stages, **backbone_kwargs),
            num_pred_layers=3,
            num_classes=num_classes,
            num_anchors=num_anchors,
        )

    @classmethod
    def D3(cls, channels_in_stages: List[int], num_classes: int, num_anchors: int, **backbone_kwargs) -> "EfficientDetObjectDetectionConfig":
        return cls(
            EfficientNetBackboneConfig.B3(channels_in_stages, **backbone_kwargs),
            num_pred_layers=4,
            num_classes=num_classes,
            num_anchors=num_anchors,
        )

    @classmethod
    def D4(cls, channels_in_stages: List[int], num_classes: int, num_anchors: int, **backbone_kwargs) -> "EfficientDetObjectDetectionConfig":
        return cls(
            EfficientNetBackboneConfig.B4(channels_in_stages, **backbone_kwargs),
            num_pred_layers=4,
            num_classes=num_classes,
            num_anchors=num_anchors,
        )

    @classmethod
    def D5(cls, channels_in_stages: List[int], num_classes: int, num_anchors: int, **backbone_kwargs) -> "EfficientDetObjectDetectionConfig":
        return cls(
            EfficientNetBackboneConfig.B5(channels_in_stages, **backbone_kwargs),
            num_pred_layers=4,
            num_classes=num_classes,
            num_anchors=num_anchors,
        )

    @classmethod
    def D6(cls, channels_in_stages: List[int], num_classes: int, num_anchors: int, **backbone_kwargs) -> "EfficientDetObjectDetectionConfig":
        return cls(
            EfficientNetBackboneConfig.B6(channels_in_stages, **backbone_kwargs),
            num_pred_layers=5,
            num_classes=num_classes,
            num_anchors=num_anchors,
        )

    @classmethod
    def D7(cls, channels_in_stages: List[int], num_classes: int, num_anchors: int, **backbone_kwargs) -> "EfficientDetObjectDetectionConfig":
        return cls(
            EfficientNetBackboneConfig.B6(channels_in_stages, **backbone_kwargs),
            num_pred_layers=5,
            num_classes=num_classes,
            num_anchors=num_anchors,
        )

    @classmethod
    def D7x(cls, channels_in_stages: List[int], num_classes: int, num_anchors: int, **backbone_kwargs) -> "EfficientDetObjectDetectionConfig":
        return cls(
            EfficientNetBackboneConfig.B7(channels_in_stages, **backbone_kwargs),
            num_pred_layers=5,
            num_classes=num_classes,
            num_anchors=num_anchors,
        )