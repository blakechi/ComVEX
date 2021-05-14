class SwinTransformerConfig(object):
    def __init__(
        self, 
        *,
        image_channel, 
        image_size, 
        patch_size,
        num_channels,
        num_layers_in_stages, 
        head_dim,
        window_size,
        shifts,
        num_classes,
        use_absolute_position,
        use_checkpoint,
        use_pre_norm=False, 
        ff_dim=None, 
        ff_dropout=0.0,
        attention_dropout=0.0,
        token_dropout=0.0,
    ) -> None:

        self.image_channel =image_channel 
        self.image_size = image_size 
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_layers_in_stages = num_layers_in_stages 
        self.head_dim = head_dim
        self.window_size = window_size
        self.shifts = shifts
        self.num_classes = num_classes
        self.use_absolute_position = use_absolute_position
        self.use_checkpoint = use_checkpoint
        self.use_pre_norm = use_pre_norm 
        self.ff_dim = ff_dim 
        self.ff_dropout = ff_dropout
        self.attention_dropout = attention_dropout
        self.token_dropout = token_dropout

    # @classmethod
    # def MLPMixer_S_32(cls, num_classes: int, ff_dropout: float = 0.0) -> "MLPMixerConfig":
    #     return cls(
    #         image_channel=3,
    #         image_size=224,
    #         patch_size=32,
    #         depth=8,
    #         token_mlp_dim=256,
    #         channel_mlp_dim=2048,
    #         num_classes=num_classes,
    #         ff_dropout=0.0
    #     )