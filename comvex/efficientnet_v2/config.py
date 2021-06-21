from typing import Optional, List, Literal

from comvex.utils import ConfigBase


class EfficientNetV2BaseConfig(ConfigBase):
    r"""
    A configuration class for `EfficientNetV2Base` only to avoid mess arguments clustering in `EfficientNetV2Config`.

    Reference from: https://github.com/google/automl/blob/master/efficientnetv2/effnetv2_configs.py#L139-L189

    According to: 
        1. https://github.com/google/automl/blob/master/efficientnetv2/effnetv2_configs.py#L214
        2. https://github.com/google/automl/blob/master/efficientnetv2/hparams.py#L226

    The last stage's `out_channel` is set to 1280 as the default and doesn't be overrided later, so we use this default for all configs.
    """

    NUM_OUT_CHANNEL_IN_LAST_CHANNEL = 1280

    def __init__(
        self,
        name: Literal["B", "S", "M", "L", "XL"],
        num_layers: List[int],
        channels: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        expand_scales: List[Optional[int]],
        se_scales: List[Optional[int]],
    ) -> None:

        self.name = name
        self.num_layers = num_layers
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.expand_scales = expand_scales
        self.se_scales = se_scales

    @classmethod
    def EfficientNetV2_B(cls) -> "EfficientNetV2BaseConfig":
        return cls(
            name="B",
            num_layers = [1, 1, 2, 2, 3, 5, 8, 1],
            channels = [32, 16, 32, 48, 96, 112, 192, cls.NUM_OUT_CHANNEL_IN_LAST_CHANNEL],
            kernel_sizes = [3]*8,
            strides = [2, 1, 2, 2, 2, 1, 2, 1],
            expand_scales = [None, 1, 4, 4, 4, 6, 6, None],
            se_scales = [*((None,)*4), *((0.25,)*3), None],
        )
        
    @classmethod
    def EfficientNetV2_S(cls) -> "EfficientNetV2BaseConfig":
        return cls(
            name="S",
            num_layers = [1, 2, 4, 4, 6, 9, 15, 1],
            channels = [24, 24, 48, 64, 128, 160, 256, cls.NUM_OUT_CHANNEL_IN_LAST_CHANNEL],  # Differ from Table 4. in the official paper
            kernel_sizes = [3]*8,
            strides = [2, 1, 2, 2, 2, 1, 2, 1],
            expand_scales = [None, 1, 4, 4, 4, 6, 6, None],
            se_scales = [*((None,)*4), *((0.25,)*3), None],
        )
        
    @classmethod
    def EfficientNetV2_M(cls) -> "EfficientNetV2BaseConfig":
        return cls(
            name="M",
            num_layers = [1, 3, 5, 5, 7, 14, 18, 5, 1],
            channels = [24, 24, 48, 80, 160, 176, 304, 512, cls.NUM_OUT_CHANNEL_IN_LAST_CHANNEL],
            kernel_sizes = [3]*9,
            strides = [2, 1, 2, 2, 2, 1, 2, 1, 1],
            expand_scales = [None, 1, 4, 4, 4, 6, 6, 6, None],
            se_scales = [*((None,)*4), *((0.25,)*4), None],
        )

    @classmethod
    def EfficientNetV2_L(cls) -> "EfficientNetV2BaseConfig":
        return cls(
            name="L",
            num_layers = [1, 4, 7, 7, 10, 19, 25, 7, 1],
            channels = [32, 32, 64, 96, 192, 224, 384, 640, cls.NUM_OUT_CHANNEL_IN_LAST_CHANNEL],
            kernel_sizes = [3]*9,
            strides = [2, 1, 2, 2, 2, 1, 2, 1, 1],
            expand_scales = [None, 1, 4, 4, 4, 6, 6, 6, None],
            se_scales = [*((None,)*4), *((0.25,)*4), None],
        )

    @classmethod
    def EfficientNetV2_XL(cls) -> "EfficientNetV2BaseConfig":
        return cls(
            name="XL",
            num_layers = [1, 4, 8, 8, 16, 24, 32, 8, 1],
            channels = [32, 32, 64, 96, 192, 256, 512, 640, cls.NUM_OUT_CHANNEL_IN_LAST_CHANNEL],
            kernel_sizes = [3]*9,
            strides = [2, 1, 2, 2, 2, 1, 2, 1, 1],
            expand_scales = [None, 1, 4, 4, 4, 6, 6, 6, None],
            se_scales = [*((None,)*4), *((0.25,)*4), None],
        )
        

class EfficientNetV2Config(ConfigBase):
    def __init__(
        self,
        base_config: EfficientNetV2BaseConfig,
        image_channel: int,
        depth_scale: float,
        width_scale: float,
        train_resolution: int,
        eval_resolution: int,
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

        self.base_config = base_config
        self.image_channel = image_channel
        self.depth_scale = depth_scale
        self.width_scale = width_scale
        self.train_resolution = train_resolution
        self.eval_resolution = eval_resolution
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

        # Add these after "Progressive Learning" done
        # self.randaug
        # self.mixup
        # self.aug

    # Reference from: https://github.com/google/automl/blob/master/efficientnetv2/effnetv2_configs.py#L190-L210
    @classmethod
    def EfficientNetV2_S(cls, num_classes: int, **kwargs) -> "EfficientNetV2Config":
        return cls(
            EfficientNetV2BaseConfig.EfficientNetV2_S(),
            3,
            1.0,
            1.0,
            300,
            384,
            num_classes=num_classes,
            ff_dropout=0.2,
            **kwargs,
        )

    @classmethod
    def EfficientNetV2_M(cls, num_classes: int, **kwargs) -> "EfficientNetV2Config":
        return cls(
            EfficientNetV2BaseConfig.EfficientNetV2_M(),
            3,
            1.0,
            1.0,
            384,
            480,
            num_classes=num_classes,
            ff_dropout=0.3,
            **kwargs,
        )

    @classmethod
    def EfficientNetV2_L(cls, num_classes: int, **kwargs) -> "EfficientNetV2Config":
        return cls(
            EfficientNetV2BaseConfig.EfficientNetV2_L(),
            3,
            1.0,
            1.0,
            384,
            480,
            num_classes=num_classes,
            ff_dropout=0.4,
            **kwargs,
        )

    @classmethod
    def EfficientNetV2_XL(cls, num_classes: int, **kwargs) -> "EfficientNetV2Config":
        return cls(
            EfficientNetV2BaseConfig.EfficientNetV2_XL(),
            3,
            1.0,
            1.0,
            384,
            512,
            num_classes=num_classes,
            ff_dropout=0.4,
            **kwargs,
        )

    @classmethod
    def EfficientNetV2_B0(cls, num_classes: int, **kwargs) -> "EfficientNetV2Config":
        return cls(
            EfficientNetV2BaseConfig.EfficientNetV2_B(),
            3,
            1.0,
            1.0,
            224,
            224,
            num_classes=num_classes,
            ff_dropout=0.2,
            **kwargs,
        )
        
    @classmethod
    def EfficientNetV2_B1(cls, num_classes: int, **kwargs) -> "EfficientNetV2Config":
        return cls(
            EfficientNetV2BaseConfig.EfficientNetV2_B(),
            3,
            1.1,
            1.0,
            240,
            240,
            num_classes=num_classes,
            ff_dropout=0.2,
            **kwargs,
        )
        
    @classmethod
    def EfficientNetV2_B2(cls, num_classes: int, **kwargs) -> "EfficientNetV2Config":
        return cls(
            EfficientNetV2BaseConfig.EfficientNetV2_B(),
            3,
            1.2,
            1.1,
            260,
            260,
            num_classes=num_classes,
            ff_dropout=0.3,
            **kwargs,
        )
        
    @classmethod
    def EfficientNetV2_B3(cls, num_classes: int, **kwargs) -> "EfficientNetV2Config":
        return cls(
            EfficientNetV2BaseConfig.EfficientNetV2_B(),
            3,
            1.4,
            1.2,
            300,
            300,
            num_classes=num_classes,
            ff_dropout=0.3,
            **kwargs,
        )