from typing import Tuple, List, Literal, Optional

from comvex.utils import ConfigBase, EfficientNetBackboneConfig, BiFPNConfig


class EfficientDetBackboneConfig(ConfigBase):
    def __init__(
        self,
        efficientnet_backbone_config: EfficientNetBackboneConfig,
        image_shapes: Tuple[int],
        bifpn_num_layers: int,
        bifpn_channel: int,
        dimension: int = 2,
        upsample_mode: Literal["nearest", "linear", "bilinear", "bicubic", "trilinear"] = "nearest",
        use_bias: bool = True,
        use_conv_after_downsampling: bool = True,
        norm_mode: Literal["fast_norm", "softmax", "channel_fast_norm", "channel_softmax"] = "fast_norm",
        batch_norm_epsilon: float = 1e-5,
        batch_norm_momentum: float = 1e-1,
        feature_map_indices: Optional[List[int]] = None
    ) -> None:
        super().__init__()

        self.efficientnet_backbone_config = efficientnet_backbone_config
        self.image_shapes = image_shapes

        bifpn_config = BiFPNConfig(
            bifpn_num_layers=bifpn_num_layers,
            bifpn_channel=bifpn_channel,
            channels_in_nodes=[],
            shapes_in_nodes=[],
            dimension=dimension,
            upsample_mode=upsample_mode,
            use_bias=use_bias,
            use_conv_after_downsampling=use_conv_after_downsampling,
            norm_mode=norm_mode,
            batch_norm_epsilon=batch_norm_epsilon,
            batch_norm_momentum=batch_norm_momentum,
        )
        for name, value in bifpn_config.__dict__.items():
            if not name in ["channels_in_nodes", "shapes_in_nodes"]:  # These will be handled automatically in `EfficientDetBackbone`
                setattr(self, name, value)
                
        self.feature_map_indices = feature_map_indices

    @classmethod
    def D0(cls, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.B0(resolution=512, strides=[1, *([2]*7), 1]),  # EfficientDet uses stride=2 from stage 2 to 8
            (512, 512),
            3,
            64,
            **kwargs
        )
        
    @classmethod
    def D1(cls, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.B1(resolution=640, strides=[1, *([2]*7), 1]),  # EfficientDet uses stride=2 from stage 2 to 8
            (640, 640),
            4,
            88,
            **kwargs
        )

    @classmethod
    def D2(cls, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.B2(resolution=768, strides=[1, *([2]*7), 1]),  # EfficientDet uses stride=2 from stage 2 to 8
            (768, 768),
            5,
            112,
            **kwargs
        )

    @classmethod
    def D3(cls, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.B3(resolution=896, strides=[1, *([2]*7), 1]),  # EfficientDet uses stride=2 from stage 2 to 8
            (896, 896),
            6,
            120,
            **kwargs
        )

    @classmethod
    def D4(cls, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.B4(resolution=1024, strides=[1, *([2]*7), 1]),  # EfficientDet uses stride=2 from stage 2 to 8
            (1024, 1024),
            7,
            224,
            **kwargs
        )

    @classmethod
    def D5(cls, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.B5(resolution=1280, strides=[1, *([2]*7), 1]),  # EfficientDet uses stride=2 from stage 2 to 8
            (1280, 1280),
            7,
            288,
            **kwargs
        )

    @classmethod
    def D6(cls, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.B6(resolution=1280, strides=[1, *([2]*7), 1]),  # EfficientDet uses stride=2 from stage 2 to 8
            (1280, 1280),
            8,
            384,
            **kwargs
        )

    @classmethod
    def D7(cls, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.B6(resolution=1536, strides=[1, *([2]*7), 1]),  # EfficientDet uses stride=2 from stage 2 to 8
            (1536, 1536),
            8,
            384,
            **kwargs
        )

    @classmethod
    def D7x(cls, **kwargs) -> "EfficientDetBackboneConfig":
        return cls(
            EfficientNetBackboneConfig.B7(resolution=1536, strides=[1, *([2]*7), 1]),  # EfficientDet uses stride=2 from stage 2 to 8
            (1536, 1536),
            8,
            384,
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