from typing import List

from comvex.utils import ConfigBase


class RandAugmentConfig(ConfigBase):
    # Partially expose arguments here, check out below for details:
    # https://github.com/tensorflow/tpu/blob/3679ca6b979349dde6da7156be2528428b000c7c/models/official/efficientnet/autoaugment.py#L470-L532
    def __init__(
        self,
        max_magnitude: int = 10,
        rotate_interval: float = 30.,  # degree
        translate_x_interval: int = 250,  # pixel
        translate_y_interval: int = 250,  # pixel
        sheer_x_interval: float = 0.3,   # portion of angle (0 ~ 180)
        sheer_y_interval: float = 0.3,   # portion of angle (0 ~ 180)
        crop_out_interval: int = 100,  # The longest lateral length (pixels) for cropping
        interpolation: str = "nearest",
        fill: float = 0.,
        addition: float = 0.,
        transform_func_distribution: List[float] = [1.]*16
    ) -> None:
        super().__init__()

        self.max_magnitude = max_magnitude
        self.rotate_interval = rotate_interval
        self.translate_x_interval = translate_x_interval
        self.translate_y_interval = translate_y_interval
        self.sheer_x_interval = sheer_x_interval
        self.sheer_y_interval = sheer_y_interval
        self.crop_out_interval = crop_out_interval
        self.interpolation = interpolation
        self.fill = fill
        self.addition = addition
        self.transform_func_distribution = transform_func_distribution
        