from typing import Callable, List, Tuple
from functools import partial

import torch
from torch import nn
try:
    from typing_extensions import Final
except:
    from torch.jit import Final

from .transfrom_functions import (
    rotate,
    translate_x,
    translate_y,
    sheer_x,
    sheer_y,
    color,
    brightness,
    sharpness,
    contrast,
    auto_contrast,
    solarize,
    solarize_add,
    posterize,
    equalize,
    invert,
    crop_out
)
from .config import RandAugmentConfig


class RandAugment(nn.Module):
    n: Final[int]
    m: Final[int]
    max_magnitude: Final[int]

    def __init__(
        self,
        n: int,
        m: int,
        config: RandAugmentConfig = RandAugmentConfig()
    ) -> None:
        super().__init__()

        self.n = n
        self.m = m
        self.max_magnitude = config.max_magnitude
        self.register_buffer("transform_func_distribution", torch.tensor(config.transform_func_distribution))

        p = (n - 1) / n  # Since we doesn't put `Identity` in the list
        self.transform_funcs: Tuple[Callable] = (
            partial(rotate, p=p, magnitude=self.m, max_magnitude=self.max_magnitude, interval=config.rotate_interval, fill=config.fill),
            partial(translate_x, p=p, magnitude=self.m, max_magnitude=self.max_magnitude, interval=config.translate_x_interval, fill=config.fill),
            partial(translate_y, p=p, magnitude=self.m, max_magnitude=self.max_magnitude, interval=config.translate_y_interval, fill=config.fill),
            partial(sheer_x, p=p, magnitude=self.m, max_magnitude=self.max_magnitude, interval=config.sheer_x_interval, fill=config.fill),
            partial(sheer_y, p=p, magnitude=self.m, max_magnitude=self.max_magnitude, interval=config.sheer_y_interval, fill=config.fill),
            partial(color, p=p, magnitude=self.m, max_magnitude=self.max_magnitude),
            partial(brightness, p=p, magnitude=self.m, max_magnitude=self.max_magnitude),
            partial(sharpness, p=p, magnitude=self.m, max_magnitude=self.max_magnitude),
            partial(contrast, p=p, magnitude=self.m, max_magnitude=self.max_magnitude),
            partial(auto_contrast, p=p),
            partial(solarize, p=p, magnitude=self.m, max_magnitude=self.max_magnitude),
            partial(solarize_add, p=p, magnitude=self.m, max_magnitude=self.max_magnitude, addition=config.addition),
            partial(posterize, p=p, magnitude=self.m, max_magnitude=self.max_magnitude),
            partial(equalize, p=p),
            partial(invert, p=p),
            partial(crop_out, p=p, magnitude=self.m, max_magnitude=self.max_magnitude, interval=config.crop_out_interval, fill=config.fill),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if not self.training:
            return x
            
        transform_funcs_indice: List[int] = self.transform_func_distribution.multinomial(num_samples=self.n).tolist()
        for idx in transform_funcs_indice:
            x = self.transform_funcs[idx](x)

        return x

    def __repr__(self):
        return f"{self.__class__.__name__}({self.n}, {self.m})"