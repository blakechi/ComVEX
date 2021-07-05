from typing import List, Dict, Optional

from comvex.utils import ConfigBase


class CoaTLiteConfig(ConfigBase):
    def __init__(
        self,
        image_size: int, 
        image_channel: int, 
        patch_size: int,
        num_layers_in_stages: List[int],
        num_channels: List[int],
        expand_scales: List[int],
        num_classes: int,
        kernel_size_on_heads: Dict[int, int] = { 3: 2, 5: 3, 7: 3 },
        heads: Optional[int] = None,
        use_bias: bool = True,
        ff_dropout=0.0,
        attention_dropout: float = 0.0,
        path_dropout: float = 0.0,
        pred_act_fnc_name: str = "ReLU",
    ) -> None:

        self.image_size = image_size
        self.image_channel = image_channel
        self.patch_size = patch_size
        self.num_layers_in_stages = num_layers_in_stages
        self.num_channels = num_channels
        self.expand_scales = expand_scales
        self.kernel_size_on_heads = kernel_size_on_heads
        self.heads = heads
        self.num_classes = num_classes
        self.use_bias = use_bias
        self.ff_dropout = ff_dropout 
        self.attention_dropout = attention_dropout
        self.path_dropout = path_dropout
        self.pred_act_fnc_name = pred_act_fnc_name

    @classmethod
    def CoaTLite_Tiny(cls, num_classes: int, **kwargs) -> "CoaTLiteConfig":
        return cls(
            224,
            3,
            4,
            [2, 2, 2, 2],
            [64, 128, 256, 320],
            [8, 8, 4, 4],
            num_classes=num_classes,
            **kwargs
        )

    @classmethod
    def CoaTLite_Mini(cls, num_classes: int, **kwargs) -> "CoaTLiteConfig":
        return cls(
            224,
            3,
            4,
            [2, 2, 2, 2],
            [64, 128, 320, 512],
            [8, 8, 4, 4],
            num_classes=num_classes,
            **kwargs
        )

    @classmethod
    def CoaTLite_Small(cls, num_classes: int, **kwargs) -> "CoaTLiteConfig":
        return cls(
            224,
            3,
            4,
            [3, 4, 6, 3],
            [64, 128, 320, 512],
            [8, 8, 4, 4],
            num_classes=num_classes,
            **kwargs
        )