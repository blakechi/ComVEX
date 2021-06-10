class MLPMixerConfig(object):
    def __init__(
        self, 
        *,
        image_channel: int,
        image_size: int,
        patch_size: int,
        depth: int,
        token_mlp_dim: int,
        channel_mlp_dim: int,
        num_classes: int,
        ff_dropout: float = 0.0,
    ) -> None:

        self.image_channel = image_channel
        self.image_size = image_size
        self.patch_size = patch_size
        self.depth = depth
        self.token_mlp_dim = token_mlp_dim
        self.channel_mlp_dim = channel_mlp_dim
        self.num_classes = num_classes
        self.ff_dropout = ff_dropout

    @classmethod
    def MLPMixer_S_32(cls, num_classes: int, **kwargs) -> "MLPMixerConfig":
        return cls(
            image_channel=3,
            image_size=224,
            patch_size=32,
            depth=8,
            token_mlp_dim=256,
            channel_mlp_dim=2048,
            num_classes=num_classes,
            **kwargs
        )

    @classmethod
    def MLPMixer_S_16(cls, num_classes: int, **kwargs) -> "MLPMixerConfig":
        return cls(
            image_channel=3,
            image_size=224,
            patch_size=16,
            depth=8,
            token_mlp_dim=256,
            channel_mlp_dim=2048,
            num_classes=num_classes,
            **kwargs
        )

    @classmethod
    def MLPMixer_B_32(cls, num_classes: int, **kwargs) -> "MLPMixerConfig":
        return cls(
            image_channel=3,
            image_size=224,
            patch_size=32,
            depth=12,
            token_mlp_dim=384,
            channel_mlp_dim=3072,
            num_classes=num_classes,
            **kwargs
        )

    @classmethod
    def MLPMixer_B_16(cls, num_classes: int, **kwargs) -> "MLPMixerConfig":
        return cls(
            image_channel=3,
            image_size=224,
            patch_size=16,
            depth=12,
            token_mlp_dim=384,
            channel_mlp_dim=3072,
            num_classes=num_classes,
            **kwargs
        )

    @classmethod
    def MLPMixer_L_32(cls, num_classes: int, **kwargs) -> "MLPMixerConfig":
        return cls(
            image_channel=3,
            image_size=224,
            patch_size=32,
            depth=24,
            token_mlp_dim=512,
            channel_mlp_dim=4096,
            num_classes=num_classes,
            **kwargs
        )

    @classmethod
    def MLPMixer_L_16(cls, num_classes: int, **kwargs) -> "MLPMixerConfig":
        return cls(
            image_channel=3,
            image_size=224,
            patch_size=32,
            depth=24,
            token_mlp_dim=512,
            channel_mlp_dim=4096,
            num_classes=num_classes,
            **kwargs
        )

    @classmethod
    def MLPMixer_H_14(cls, num_classes: int, **kwargs) -> "MLPMixerConfig":
        return cls(
            image_channel=3,
            image_size=224,
            patch_size=14,
            depth=32,
            token_mlp_dim=640,
            channel_mlp_dim=5120,
            num_classes=num_classes,
            **kwargs
        )
    