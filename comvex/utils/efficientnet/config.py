from typing import Optional

from comvex.utils import ConfigBase


class EfficientNetBackboneConfig(ConfigBase):
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
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.99,
        return_feature_maps: bool = False,
        path_dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.image_channel = image_channel
        self.depth_scale = depth_scale
        self.width_scale = width_scale
        self.resolution = resolution
        self.up_sampling_mode = up_sampling_mode
        self.act_fnc_name = act_fnc_name
        self.se_act_fnc_name = se_act_fnc_name
        self.se_scale = se_scale
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum
        self.return_feature_maps = return_feature_maps
        self.path_dropout = path_dropout

    # Reference from: https://github.com/tensorflow/tpu/blob/3679ca6b979349dde6da7156be2528428b000c7c/models/official/efficientnet/efficientnet_builder.py#L39-L48
    @classmethod
    def B0(cls, **kwargs) -> "EfficientNetBackboneConfig":
        return cls(
            3,
            1.0,
            1.0,
            resolution=kwargs["resolution"] if hasattr(kwargs, "resolution") else 224,
            **kwargs,
        )
        
    @classmethod
    def B1(cls, **kwargs) -> "EfficientNetBackboneConfig":
        return cls(
            3,
            1.1,
            1.0,
            resolution=kwargs["resolution"] if hasattr(kwargs, "resolution") else 240,
            **kwargs,
        )
        
    @classmethod
    def B2(cls, **kwargs) -> "EfficientNetBackboneConfig":
        return cls(
            3,
            1.2,
            1.1,
            resolution=kwargs["resolution"] if hasattr(kwargs, "resolution") else 260,
            **kwargs,
        )
        
    @classmethod
    def B3(cls, **kwargs) -> "EfficientNetBackboneConfig":
        return cls(
            3,
            1.4,
            1.2,
            resolution=kwargs["resolution"] if hasattr(kwargs, "resolution") else 300,
            **kwargs,
        )
        
    @classmethod
    def B4(cls, **kwargs) -> "EfficientNetBackboneConfig":
        return cls(
            3,
            1.8,
            1.4,
            resolution=kwargs["resolution"] if hasattr(kwargs, "resolution") else 380,
            **kwargs,
        )
        
    @classmethod
    def B5(cls, **kwargs) -> "EfficientNetBackboneConfig":
        return cls(
            3,
            2.2,
            1.6,
            resolution=kwargs["resolution"] if hasattr(kwargs, "resolution") else 456,
            **kwargs,
        )
        
    @classmethod
    def B6(cls, **kwargs) -> "EfficientNetBackboneConfig":
        return cls(
            3,
            2.6,
            1.8,
            resolution=kwargs["resolution"] if hasattr(kwargs, "resolution") else 528,
            **kwargs,
        )
        
    @classmethod
    def B7(cls, **kwargs) -> "EfficientNetBackboneConfig":
        return cls(
            3,
            3.1,
            2.0,
            resolution=kwargs["resolution"] if hasattr(kwargs, "resolution") else 600,
            **kwargs,
        )
        
    @classmethod
    def B8(cls, **kwargs) -> "EfficientNetBackboneConfig":
        return cls(
            3,
            3.6,
            2.2,
            resolution=kwargs["resolution"] if hasattr(kwargs, "resolution") else 672,
            **kwargs,
        )
        
    @classmethod
    def L2(cls, **kwargs) -> "EfficientNetBackboneConfig":
        return cls(
            3,
            5.3,
            4.3,
            resolution=kwargs["resolution"] if hasattr(kwargs, "resolution") else 800,
            **kwargs,
        )


class EfficientNetConfig(ConfigBase):
    def __init__(
        self,
        image_channel: int,
        depth_scale: float,
        width_scale: float,
        resolution: int,
        num_classes: int,
        up_sampling_mode: Optional[str] = None,
        act_fnc_name: str = "SiLU",
        se_act_fnc_name: str = "SiLU",
        se_scale: float = 0.25,
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.99,
        return_feature_maps: bool = False,
        path_dropout: float = 0.2,
        ff_dropout: float = 0.,
    ) -> None:
        super().__init__()

        self.image_channel = image_channel
        self.depth_scale = depth_scale
        self.width_scale = width_scale
        self.resolution = resolution
        self.num_classes = num_classes
        self.up_sampling_mode = up_sampling_mode
        self.act_fnc_name = act_fnc_name
        self.se_act_fnc_name = se_act_fnc_name
        self.se_scale = se_scale
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum
        self.return_feature_maps = return_feature_maps
        self.path_dropout = path_dropout
        self.ff_dropout = ff_dropout

    # Reference from: https://github.com/tensorflow/tpu/blob/3679ca6b979349dde6da7156be2528428b000c7c/models/official/efficientnet/efficientnet_builder.py#L39-L48
    @classmethod
    def EfficientNet_B0(cls, num_classes: int, **kwargs) -> "EfficientNetConfig":
        return cls(
            3,
            1.0,
            1.0,
            resolution=kwargs["resolution"] if hasattr(kwargs, "resolution") else 224,
            num_classes=num_classes,
            ff_dropout=0.2,
            **kwargs,
        )
        
    @classmethod
    def EfficientNet_B1(cls, num_classes: int, **kwargs) -> "EfficientNetConfig":
        return cls(
            3,
            1.1,
            1.0,
            resolution=kwargs["resolution"] if hasattr(kwargs, "resolution") else 240,
            num_classes=num_classes,
            ff_dropout=0.2,
            **kwargs,
        )
        
    @classmethod
    def EfficientNet_B2(cls, num_classes: int, **kwargs) -> "EfficientNetConfig":
        return cls(
            3,
            1.2,
            1.1,
            resolution=kwargs["resolution"] if hasattr(kwargs, "resolution") else 260,
            num_classes=num_classes,
            ff_dropout=0.3,
            **kwargs,
        )
        
    @classmethod
    def EfficientNet_B3(cls, num_classes: int, **kwargs) -> "EfficientNetConfig":
        return cls(
            3,
            1.4,
            1.2,
            resolution=kwargs["resolution"] if hasattr(kwargs, "resolution") else 300,
            num_classes=num_classes,
            ff_dropout=0.3,
            **kwargs,
        )
        
    @classmethod
    def EfficientNet_B4(cls, num_classes: int, **kwargs) -> "EfficientNetConfig":
        return cls(
            3,
            1.8,
            1.4,
            resolution=kwargs["resolution"] if hasattr(kwargs, "resolution") else 380,
            num_classes=num_classes,
            ff_dropout=0.4,
            **kwargs,
        )
        
    @classmethod
    def EfficientNet_B5(cls, num_classes: int, **kwargs) -> "EfficientNetConfig":
        return cls(
            3,
            2.2,
            1.6,
            resolution=kwargs["resolution"] if hasattr(kwargs, "resolution") else 456,
            num_classes=num_classes,
            ff_dropout=0.4,
            **kwargs,
        )
        
    @classmethod
    def EfficientNet_B6(cls, num_classes: int, **kwargs) -> "EfficientNetConfig":
        return cls(
            3,
            2.6,
            1.8,
            resolution=kwargs["resolution"] if hasattr(kwargs, "resolution") else 528,
            num_classes=num_classes,
            ff_dropout=0.5,
            **kwargs,
        )
        
    @classmethod
    def EfficientNet_B7(cls, num_classes: int, **kwargs) -> "EfficientNetConfig":
        return cls(
            3,
            3.1,
            2.0,
            resolution=kwargs["resolution"] if hasattr(kwargs, "resolution") else 600,
            num_classes=num_classes,
            ff_dropout=0.5,
            **kwargs,
        )
        
    @classmethod
    def EfficientNet_B8(cls, num_classes: int, **kwargs) -> "EfficientNetConfig":
        return cls(
            3,
            3.6,
            2.2,
            resolution=kwargs["resolution"] if hasattr(kwargs, "resolution") else 672,
            num_classes=num_classes,
            ff_dropout=0.5,
            **kwargs,
        )
        
    @classmethod
    def EfficientNet_L2(cls, num_classes: int, **kwargs) -> "EfficientNetConfig":
        return cls(
            3,
            5.3,
            4.3,
            resolution=kwargs["resolution"] if hasattr(kwargs, "resolution") else 800,
            num_classes=num_classes,
            ff_dropout=0.5,
            **kwargs,
        )
        
