from comvex.utils import ConfigBase


class ViPConfig(ConfigBase):
    def __init__(
        self,
        image_channel: int,
        image_size: int,            # one lateral's size of a squre image
        patch_size: int,            # one lateral's size of a squre patch
        layers_in_stages: int,
        channels_in_stages: int,
        num_classes: int,
        use_weighted: bool = True,  # Whether to use `Weighted Permute - MLP` or `Permute - MLP`
        use_bias: bool = False,
        ff_dropout: float = 0.,
        path_dropout: float = 0.,
    ) -> None:
        super().__init__()

        self.image_channel=image_channel
        self.image_size=image_size
        self.patch_size=patch_size
        self.layers_in_stages=layers_in_stages
        self.channels_in_stages=channels_in_stages
        self.num_classes=num_classes
        self.use_weighted=use_weighted
        self.use_bias=use_bias
        self.ff_dropout=ff_dropout
        self.path_dropout=path_dropout

    @classmethod
    def ViP_Small_14(cls, num_classes: int, **kwargs) -> "ViPConfig":
        return cls(
            3,
            224,
            patch_size=14,
            layers_in_stages=[4, 3, 8, 3],
            channels_in_stages=[384, 384, 384, 384],
            num_classes=num_classes,
            path_dropout=0.1,
            **kwargs
        )

    @classmethod
    def ViP_Small_7(cls, num_classes: int, **kwargs) -> "ViPConfig":
        return cls(
            3,
            224,
            patch_size=7,
            layers_in_stages=[4, 3, 8, 3],
            channels_in_stages=[384, 384, 384, 384],
            num_classes=num_classes,
            path_dropout=0.1,
            **kwargs
        )

    @classmethod
    def ViP_Medium_7(cls, num_classes: int, **kwargs) -> "ViPConfig":
        return cls(
            3,
            224,
            patch_size=7,
            layers_in_stages=[4, 3, 14, 3],
            channels_in_stages=[256, 256, 512, 512],
            num_classes=num_classes,
            path_dropout=0.1,
            **kwargs
        )

    @classmethod
    def ViP_Large_7(cls, num_classes: int, **kwargs) -> "ViPConfig":
        return cls(
            3,
            224,
            patch_size=7,
            layers_in_stages=[8, 8, 16, 4],
            channels_in_stages=[256, 256, 512, 512],
            num_classes=num_classes,
            path_dropout=0.1,
            **kwargs
        )