from typing import List, Union

from comvex.utils import ConfigBase


class CoAtNetConfig(ConfigBase):
    def __init__(
        self,
        image_height: int,
        image_width: int,
        image_channel: int,
        num_blocks_in_layers: List[int],
        block_type_in_layers: List[str],
        num_channels_in_layers: Union[List[int], int],
        expand_scale_in_layers: Union[List[int], int],
        heads: int,
        num_classes: int,
        pred_act_fnc_name: str = "ReLU",
        ff_dropout: float = 0.,
        attention_dropout: float = 0.,
        path_dropout: float = 0.,    
    ) -> None:

        self.image_height = image_height
        self.image_width = image_width
        self.image_channel = image_channel
        self.num_blocks_in_layers = num_blocks_in_layers
        self.block_type_in_layers = block_type_in_layers
        self.num_channels_in_layers = num_channels_in_layers
        self.expand_scale_in_layers = expand_scale_in_layers
        self.heads = heads
        self.num_classes = num_classes
        self.pred_act_fnc_name = pred_act_fnc_name
        self.ff_dropout = ff_dropout
        self.attention_dropout = attention_dropout
        self.path_dropout = path_dropout

    @classmethod
    def CoAtNet_0(cls, num_classes: int, **kwargs) -> 'CoAtNetConfig':
        return cls(
            224,
            224,
            3,
            [2, 2, 3, 5, 2],
            ['C', 'C', 'T', 'T'],
            [64, 96, 192, 384, 768],
            4,
            32,
            num_classes=num_classes,
            **kwargs
        )
    
    @classmethod
    def CoAtNet_1(cls, num_classes: int, **kwargs) -> 'CoAtNetConfig':
        return cls(
            224,
            224,
            3,
            [2, 2, 6, 14, 2],
            ['C', 'C', 'T', 'T'],
            [64, 96, 192, 384, 768],
            4,
            32,
            num_classes=num_classes,
            **kwargs
        )

    @classmethod
    def CoAtNet_2(cls, num_classes: int, **kwargs) -> 'CoAtNetConfig':
        return cls(
            224,
            224,
            3,
            [2, 2, 6, 14, 2],
            ['C', 'C', 'T', 'T'],
            [128, 128, 256, 512, 1024],
            4,
            32,
            num_classes=num_classes,
            **kwargs
        )

    @classmethod
    def CoAtNet_3(cls, num_classes: int, **kwargs) -> 'CoAtNetConfig':
        return cls(
            224,
            224,
            3,
            [2, 2, 6, 14, 2],
            ['C', 'C', 'T', 'T'],
            [192, 192, 384, 768, 1536],
            4,
            32,
            num_classes=num_classes,
            **kwargs
        )

    @classmethod
    def CoAtNet_4(cls, num_classes: int, **kwargs) -> 'CoAtNetConfig':
        return cls(
            224,
            224,
            3,
            [2, 2, 12, 28, 2],
            ['C', 'C', 'T', 'T'],
            [192, 192, 384, 768, 1536],
            4,
            32,
            num_classes=num_classes,
            **kwargs
        )