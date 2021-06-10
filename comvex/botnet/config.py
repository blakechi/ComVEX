from typing import Optional

from comvex.utils import ResNetConfig


class BoTNetConfig(ResNetConfig):
    def __init__(
        self, 
        num_input_channel: int,
        input_lateral_size: int,
        conv_base_block_name: str,
        bot_base_block_name: str,
        num_blocks_in_layer: list, 
        num_heads: int = 4,
        bot_block_indicator: list = [1, 1, 1],
        num_classes: Optional[int] = None,
    ) -> None:

        assert num_input_channel > 0, self._decorate_message(
             "`num_input_channel` should be specified and greater than 0"
        )
        assert input_lateral_size > 0, self._decorate_message(
             "`input_lateral_size` must be greater than 0"
        )
        assert conv_base_block_name in BoTNetConfig.available_conv_base_blocks(), self._decorate_message(
            f"`conv_base_block_name` ({conv_base_block_name}) should be one as listed below: \n{BoTNetConfig.available_conv_base_blocks()}"
        )
        assert bot_base_block_name in BoTNetConfig.available_bot_base_blocks(), self._decorate_message(
            f"`bot_base_block_name` ({bot_base_block_name}) should be one as listed below: \n{BoTNetConfig.available_bot_base_blocks()}"
        )
        assert len(num_blocks_in_layer) == 4, self._decorate_message(
             "The length of `num_blocks_in_layer` must be qual to 4 for conv_2 to conv_4 plus one BoT layer"
        )
        assert num_heads > 0, self._decorate_message(
             "`num_heads` must be greater than 0"
        )
        assert len(bot_block_indicator) == num_blocks_in_layer[-1], self._decorate_message(
             "The length of `bot_block_indicator` must be equal to the number of blocks specified in the last element of `num_blocks_in_layer`"
        )

        if num_classes is not None: 
            assert num_classes > 0, self._decorate_message(
                "`num_classes` should be specified and greater than 0"
            )

        self.num_input_channel = num_input_channel
        self.input_lateral_size = input_lateral_size
        self.conv_base_block_name = conv_base_block_name
        self.bot_base_block_name = bot_base_block_name

        # make code more readable
        layer_keys = [f"conv_{idx}" for idx in range(2, 4 + 1)] + ["bot"]
        self.num_blocks_in_layer = dict(zip(layer_keys, num_blocks_in_layer))

        self.num_heads = num_heads
        self.bot_block_indicator = bot_block_indicator
        self.num_classes = num_classes

    @staticmethod
    def available_conv_base_blocks() -> list: 
        return [
            "ResNetBlock", 
            "ResNetBottleneckBlock", 
            "ResNetFullPreActivationBlock", 
            "ResNetFullPreActivationBottleneckBlock"
        ]

    @staticmethod
    def available_bot_base_blocks() -> list: 
        return [
            "BoTNetBlock",
            "BoTNetFullPreActivationBlock"
        ]

    @classmethod
    def BoTNet_50(cls, **kwargs):
        return cls(
            3,
            1024,
            "ResNetFullPreActivationBottleneckBlock",
            "BoTNetFullPreActivationBlock",
            [3, 4, 6, 3],
            **kwargs
        )

    @classmethod
    def BoTNet_50_ImageNet(cls):
        return cls(
            3,
            1024,
            "ResNetFullPreActivationBottleneckBlock",
            "BoTNetFullPreActivationBlock",
            [3, 4, 6, 3],
            num_classes=1000
        )