from typing import Tuple

import torch
from torch import nn
try:
    from typing_extensions import Final
except:
    from torch.jit import Final

from comvex.utils import RandAugmentConfig, RandAugment, MixUp


class ProgressiveLearning(nn.Module):

    num_total_steps: Final[int]
    num_stages: Final[int]
    steps_per_stage: Final[int]
    rand_aug_transform_steps: Final[int]

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        n: int,
        m: int,
        rand_aug_transform_steps: int,
        init_image_size: int,
        target_image_size: int,
        init_magnitude: float,
        target_magnitude: float,
        init_alpha: float,
        target_alpha: float,
        init_dropout: float,
        target_dropout: float,
    ) -> None:
        super().__init__()

        # Model and Loss Function
        self.model = model
        self.criterion = criterion

        # Progressive Learning
        self.num_total_steps = n
        self.num_stages = m
        self.steps_per_stage = n // m
        self.curr_stage = 0
        self.rand_aug_transform_steps = rand_aug_transform_steps

        # Init
        # [initial, current, target]
        self.image_size_range = [init_image_size, init_image_size, target_image_size]
        self.magnitude_range = [init_magnitude, init_magnitude, target_magnitude]
        self.alpha_range = [init_alpha, init_alpha, target_alpha]
        self.dropout_range = [init_dropout, init_dropout, target_dropout]

        # RandAugment
        self.rand_augment = RandAugment(
            self.rand_aug_transform_steps,
            self.init_magnitude,
            RandAugmentConfig(max_magnitude=self.target_magnitude)
        )

        # mixup
        self.mix_up = MixUp(init_alpha)

    def forward(self, x: torch.Tensor, y: torch.Tensor, curr_step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if curr_step > 0 and curr_step % self.steps_per_stage == 0:
            self.update()

        x = self.rand_augment(x)
        x = self.mix_up(x, y)

        y_logit = self.model(x)
        loss = self.mix_up.get_loss(self.criterion, y_logit)

        return (y_logit, loss)

    def update(self) -> None:
        self.curr_stage += 1

        self.update_factors_(self.image_size_range, to_int=True)
        self.update_factors_(self.magnitude_range)
        self.update_factors_(self.alpha_range)
        self.update_factors_(self.dropout_range)
        
        self.rand_augment = RandAugment(
            self.rand_aug_transform_steps,
            self.magnitude_range[1],
            RandAugmentConfig(max_magnitude=self.target_magnitude)
        )
        self.mix_up = MixUp(self.alpha_range[1])

    def update_factors_(self, factor_range, to_int: bool) -> None:
        initial, current, target = factor_range

        current = initial + (target - initial)*(self.curr_stages / (self.num_stages - 1))
        current = int(current) if to_int else current

        factor_range[1] = current
