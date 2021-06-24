from typing import Tuple

import torch
from torch import nn


class MixUp(nn.Module):
    r"""
    Implementation of mixup: BEYOND EMPIRICAL RISK MINIMIZATION (https://arxiv.org/abs/1710.09412)
    Official implementation: https://github.com/facebookresearch/mixup-cifar10

    Note: Can sit inside a model as a method or be used in the training loop without initialization
    """

    def __init__(self, alpha: float) -> None:
        super().__init__()

        self.alpha = alpha

        # Cache
        self.lambda_scale = -1.
        self.y = torch.empty()
        self.y_perm = torch.empty()

    @property
    def lambda_scale(self) -> float:
        return self.lambda_scale

    @property
    def y(self) -> torch.Tensor:
        return self.y

    @property
    def y_perm(self) -> torch.Tensor:
        return self.y_perm

    def reset_cache(self) -> None:
        self.lambda_scale = -1.
        self.y = torch.empty()
        self.y_perm = torch.empty()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        r"""
        x (b, ...): a minibatch of input data from dataset
        y (b, ...): a minibatch of labels from dataset
        """

        out, self.y, self.y_perm, self.lambda_scale = self.mix(x, y, self.alpha, self.training)
        
        return out

    def get_loss(self, criterion: nn.Module, y_pred: torch.Tensor) -> torch.Tensor:
        r"""
        Don't forget to call this function after forward propagations.
        Will reset the cache after getting the loss.
        """

        loss = self.loss(criterion, y_pred, self.y, self.y_perm, self.lambda_scale)
        self.reset_cache()

        return loss

    @staticmethod
    def mix(x: torch.Tensor, y: torch.Tensor, alpha: float, is_training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        r"""
        x (b, ...): torch.Tensor - A minibatch of input data from dataset
        y (b, ...): torch.Tensor - A minibatch of labels from dataset
        alpha: float
        is_training: bool - Default: True
        """

        if not is_training:
            return (x, y)

        b: int = x.shape[0]

        perm_indices: torch.Tensor = torch.randperm(b, device=x.device)
        lambda_scale: float = torch.distributions.beta(alpha, alpha) if alpha > 0. else 1.

        x = lambda_scale*x + (1. - lambda_scale)*x.index_select(dim=0, index=perm_indices)
        y_perm = y.index_select(dim=0, index=perm_indices)

        return (x, y, y_perm, lambda_scale)

    @staticmethod
    def loss(
        criterion: nn.Module,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        y_perm: torch.Tensor,
        lambda_scale: float
    ) -> torch.Tensor:
        return lambda_scale*criterion(y_pred, y) + (1. - lambda_scale)*criterion(y_pred, y_perm)