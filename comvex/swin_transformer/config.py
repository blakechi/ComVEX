from typing import List

class SwinTransformerConfig(object):
    def __init__(
        self, 
        image_channel: int, 
        image_size: int, 
        patch_size: int,
        num_channels: int,
        num_layers_in_stages: List[int], 
        head_dim: int,
        window_size: int,
        shifts: int,
        num_classes: int,
        use_absolute_position: bool = False,
        use_checkpoint: bool = False,
        use_pre_norm: bool = False, 
        ff_dim: int = None, 
        ff_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        token_dropout: float = 0.0, 
        path_dropout: float = 0.0,
        pred_act_fnc_name: str = "ReLU"
    ) -> None:

        self.image_channel = image_channel 
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
        self.path_dropout = path_dropout
        self.pred_act_fnc_name = pred_act_fnc_name

    @classmethod
    def SwinTransformer_T(cls, num_classes: int, **kwargs) -> "SwinTransformerConfig":
        return cls(
            image_channel=3, 
            image_size=224, 
            patch_size=4,
            num_channels=96,
            num_layers_in_stages=[2, 2, 6, 2], 
            head_dim=32,
            window_size=(7, 7),
            shifts=2,
            num_classes=num_classes,
            **kwargs
        )

    @classmethod
    def SwinTransformer_S(cls, num_classes: int, **kwargs) -> "SwinTransformerConfig":
        return cls(
            image_channel=3, 
            image_size=224, 
            patch_size=4,
            num_channels=96,
            num_layers_in_stages=[2, 2, 18, 2], 
            head_dim=32,
            window_size=(7, 7),
            shifts=2,
            num_classes=num_classes,
            **kwargs
        )
    
    @classmethod
    def SwinTransformer_B(cls, num_classes: int, **kwargs) -> "SwinTransformerConfig":
        return cls(
            image_channel=3, 
            image_size=224, 
            patch_size=4,
            num_channels=128,
            num_layers_in_stages=[2, 2, 18, 2], 
            head_dim=32,
            window_size=(7, 7),
            shifts=2,
            num_classes=num_classes,
            **kwargs
        )

    @classmethod
    def SwinTransformer_L(cls, num_classes: int, **kwargs) -> "SwinTransformerConfig":
        return cls(
            image_channel=3, 
            image_size=224, 
            patch_size=4,
            num_channels=192,
            num_layers_in_stages=[2, 2, 18, 2], 
            head_dim=32,
            window_size=(7, 7),
            shifts=2,
            num_classes=num_classes,
            **kwargs
        )