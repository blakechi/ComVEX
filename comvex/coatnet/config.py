from typing import List, Union

from comvex.utils import ConfigBase


class CoAtNetConfig(ConfigBase):
    def __init__(
        self,
        image_height: int,
        image_width: int,
        image_channel: int,
        num_blocks_in_layers: List[int],
        block_type_in_layers: List[int],
        num_channels_in_layers: Union[List[int], int],
        expand_scale_in_layers: Union[List[int], int],
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
        self.num_classes = num_classes
        self.pred_act_fnc_name = pred_act_fnc_name
        self.ff_dropout = ff_dropout
        self.attention_dropout = attention_dropout
        self.path_dropout = path_dropout