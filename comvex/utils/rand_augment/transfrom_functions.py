from typing import Callable
from numbers import Number

import functools

import torch
from torchvision.transforms import functional as TF


def _to_negative(factor: int) -> int:
    return -factor if torch.rand(1) > 0.5 else factor


def _random_apply(transform_func: Callable) -> Callable:

    @functools.wraps(transform_func)
    def wrapper(*args, **kwargs):
        p = kwargs.pop("p")

        return transform_func(*args, **kwargs) if torch.rand(1) < p else args[0]

    return wrapper


@_random_apply
def rotate(
    x: torch.Tensor,
    interval: float,  # [0, interval]
    magnitude: float,
    max_magnitude: int,
    fill: float,
) -> torch.Tensor:
    
    degree = (magnitude / max_magnitude)*interval
    degree = _to_negative(degree)
    
    return TF.rotate(x, degree, fill=[fill,])


@_random_apply
def translate_x(
    x: torch.Tensor,
    interval: float,  # [0, interval]
    magnitude: float,
    max_magnitude: int,
    fill: float,
) -> torch.Tensor:

    # Differ from: https://github.com/tensorflow/tpu/blob/3679ca6b979349dde6da7156be2528428b000c7c/models/official/efficientnet/autoaugment.py#L503-L507
    pixels = int((magnitude / max_magnitude)*interval)
    pixels = _to_negative(pixels)

    return TF.affine(x, angle=0., translate=[pixels, 0], shear=[0., 0.], scale=1., fill=[fill,])


@_random_apply
def translate_y(
    x: torch.Tensor,
    interval: float,  # [0, interval]
    magnitude: float,
    max_magnitude: int,
    fill: float,
) -> torch.Tensor:

    # Differ from: https://github.com/tensorflow/tpu/blob/3679ca6b979349dde6da7156be2528428b000c7c/models/official/efficientnet/autoaugment.py#L503-L507
    pixels = int((magnitude / max_magnitude)*interval)
    pixels = _to_negative(pixels)

    return TF.affine(x, angle=0., translate=[0, fraction], shear=[0., 0.], scale=1., fill=[fill,])


@_random_apply
def sheer_x(
    x: torch.Tensor,
    interval: float,  # [0, interval]
    magnitude: float,
    max_magnitude: int,
    fill: float,
) -> torch.Tensor:

    angle = 180*(magnitude / max_magnitude)*interval
    angle = _to_negative(angle)

    return TF.affine(x, angle=0., translate=[0, 0], shear=[angle, 0.], scale=1., fill=[fill,])


@_random_apply
def sheer_y(
    x: torch.Tensor,
    interval: float,  # [0, interval]
    magnitude: float,
    max_magnitude: int,
    fill: float,
) -> torch.Tensor:

    angle = 180*(magnitude / max_magnitude)*interval
    angle = _to_negative(angle)

    return TF.affine(x, angle=0., translate=[0, 0], shear=[0., angle], scale=1., fill=[fill,])


@_random_apply
def color(
    x: torch.Tensor,
    magnitude: float,
    max_magnitude: int,
) -> torch.Tensor:
    
    # From: https://github.com/tensorflow/tpu/blob/3679ca6b979349dde6da7156be2528428b000c7c/models/official/efficientnet/autoaugment.py#L492
    factor = 1.8*(magnitude / max_magnitude) + 0.1

    return TF.adjust_saturation(x, factor)


@_random_apply
def brightness(x: torch.Tensor,
    magnitude: float,
    max_magnitude: int,
) -> torch.Tensor:
    
    # From: https://github.com/tensorflow/tpu/blob/3679ca6b979349dde6da7156be2528428b000c7c/models/official/efficientnet/autoaugment.py#L492
    factor = 1.8*(magnitude / max_magnitude) + 0.1

    return TF.adjust_brightness(x, factor)


@_random_apply
def sharpness(x: torch.Tensor,
    magnitude: float,
    max_magnitude: int,
) -> torch.Tensor:
    
    # From: https://github.com/tensorflow/tpu/blob/3679ca6b979349dde6da7156be2528428b000c7c/models/official/efficientnet/autoaugment.py#L492
    factor = 1.8*(magnitude / max_magnitude) + 0.1

    return TF.adjust_sharpness(x, factor)


@_random_apply
def contrast(x: torch.Tensor,
    magnitude: float,
    max_magnitude: int,
) -> torch.Tensor:
    
    # From: https://github.com/tensorflow/tpu/blob/3679ca6b979349dde6da7156be2528428b000c7c/models/official/efficientnet/autoaugment.py#L492
    factor = 1.8*(magnitude / max_magnitude) + 0.1

    return TF.adjust_contrast(x, factor)


@_random_apply
def auto_contrast(x: torch.Tensor) -> torch.Tensor:
    return TF.autocontrast(x)


@_random_apply
def solarize(
    x: torch.Tensor,
    magnitude: float,
    max_magnitude: int,
) -> torch.Tensor:
    
    threshold = (magnitude / max_magnitude)*1.0

    return TF.solarize(x, threshold=threshold)


@_random_apply
def solarize_add(
    x: torch.Tensor,
    magnitude: float,
    max_magnitude: int,
    addition: float,
) -> torch.Tensor:

    # From: https://github.com/tensorflow/tpu/blob/3679ca6b979349dde6da7156be2528428b000c7c/models/official/efficientnet/autoaugment.py#L183
    # From: https://github.com/tensorflow/tpu/blob/3679ca6b979349dde6da7156be2528428b000c7c/models/official/efficientnet/autoaugment.py#L518
    threshold = (magnitude / max_magnitude)*0.4297  # 110 / 256

    added_x = x + addition
    x = torch.where(x < threshold, added_x.clamp(min=0., max=1.), x)

    return x


@_random_apply
def posterize(
    x: torch.Tensor,
    magnitude: float,
    max_magnitude: int,
) -> torch.Tensor:

    # From: https://github.com/tensorflow/tpu/blob/3679ca6b979349dde6da7156be2528428b000c7c/models/official/efficientnet/autoaugment.py#L516
    bits = int((magnitude / max_magnitude)*4)
    dtype = x.dtype
    x = x.to(torch.uint8)

    return TF.posterize(x, bits=bits).to(dtype)


@_random_apply
def equalize(x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x = x.to(torch.uint8)

    return TF.equalize(x).to(dtype)


@_random_apply
def invert(x: torch.Tensor) -> torch.Tensor:
    return TF.invert(x)


@_random_apply
def crop_out(
    x: torch.Tensor,
    interval: int,
    magnitude: float,
    max_magnitude: int,
    fill: float,
) -> torch.Tensor:

    # From: https://github.com/tensorflow/tpu/blob/3679ca6b979349dde6da7156be2528428b000c7c/models/official/efficientnet/autoaugment.py#L125
    H, W = x.shape[-2], x.shape[-1]

    # From: https://github.com/tensorflow/tpu/blob/3679ca6b979349dde6da7156be2528428b000c7c/models/official/efficientnet/autoaugment.py#L525
    # From: https://github.com/tensorflow/tpu/blob/3679ca6b979349dde6da7156be2528428b000c7c/models/official/efficientnet/autoaugment.py#L666-L667
    lateral_len = int((magnitude / max_magnitude)*interval)  # Default: interval = 100

    x_coor = torch.randint(0, W - lateral_len, []).item()
    y_coor = torch.randint(0, H - lateral_len, []).item()

    return TF.erase(x, i=x_coor, j=y_coor, h=lateral_len, w=lateral_len, v=fill)
