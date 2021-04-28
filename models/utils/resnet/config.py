from typing import Optional


class ResNetConfig(object):
    def __init__(
        self, 
        num_input_channel: int,
        base_block_name: str, 
        num_blocks_in_conv_layer: list, 
        *,
        num_classes: Optional[int] = None,
    ) -> None:

        assert num_input_channel > 0, self._decorate_message(
             "`num_input_channel` should be specified and greater than 0"
        )
        assert base_block_name in ResNetConfig.available_base_blocks(), self._decorate_message(
            f"`base_block_name` ({base_block_name}) should be one as listed below: \n{ResNetConfig.available_base_blocks()}"
        )
        assert len(num_blocks_in_conv_layer) == 4, self._decorate_message(
             "The length of `num_blocks_in_conv_layer` must be qual to 4 for conv_2 to conv_5"
        )

        if num_classes is not None: 
            assert num_classes > 0, self._decorate_message(
                "`num_classes` should be specified and greater than 0"
            )

        self.num_input_channel = num_input_channel
        self.base_block_name = base_block_name

        # make code more readable
        conv_keys = [f"conv_{idx}" for idx in range(2, 5 + 1)]
        self.num_blocks_in_conv_layer = dict(zip(conv_keys, num_blocks_in_conv_layer))

        self.num_classes = num_classes

    @staticmethod
    def available_base_blocks() -> list: 
        return [
            "ResNetBlock", 
            "ResNetBottleneckBlock", 
            "ResNetFullPreActivationBlock", 
            "ResNetFullPreActivationBottleneckBlock"
        ]

    def _decorate_message(self, msg: str = ""):
        return f"[{self.__class__.__name__}] {msg}"

    @classmethod
    def ResNet_50(cls):
        return cls(
            3,
            "ResNetFullPreActivationBottleneckBlock",
            [3, 4, 6, 3],
        )

    @classmethod
    def ResNet_50_ImageNet(cls):
        return cls(
            3,
            "ResNetFullPreActivationBottleneckBlock",
            [3, 4, 6, 3],
            num_classes=1000
        )