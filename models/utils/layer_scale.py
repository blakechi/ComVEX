from typing import Optional, Union
import torch
from torch import nn

from .dropout import PathDropout


class AffineTransform(nn.Module):
    r"""
    Affine Transformation from ResMLP: https://arxiv.org/abs/2105.03404

    Note: 
        - Using defaults for pre-normalization Aff.
        - Setting `alpha` to a small value depending on the depth of your networks and `beta` to None for post-normalization Aff.
    """
    def __init__(self, dim: int, alpha: float = 1., beta: Optional[float] = 0.) -> None:
        super().__init__()

        self.alpha = nn.Parameter(alpha*torch.ones(dim), requires_grad=True)
        self.beta = nn.Parameter(beta*torch.ones(dim), requires_grad=True) if beta is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha*x + self.beta if self.beta is not None else self.alpha*x


class LayerScaleBlock(AffineTransform):
    r"""
    Layer Scale from CaiT (Figure 1 (d)): https://arxiv.org/abs/2103.17239

    Note: We replace `lambda` used in the official paper with `alpha`
    """
    def __init__(
        self, 
        dim: int, 
        core_block: Union[nn.Module, str], 
        pre_norm: Union[nn.Module, str] = "LayerNorm", 
        alpha: float = 1e-4, 
        path_dropout=0., 
        **kwargs  # kwargs for the `core_block`
    ) -> None:
        super().__init__(dim, alpha=alpha, beta=None)

        self.core = nn.Sequential(
            pre_norm(dim) if issubclass(pre_norm, nn.Module) else getattr(nn, pre_norm)(dim),
            core_block(dim, **kwargs) if issubclass(core_block, nn.Module) else getattr(nn, core_block)(dim, **kwargs)
        )
        self.path_dropout = PathDropout(path_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.path_dropout(super().forward(self.core(x)))
        