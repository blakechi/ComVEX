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


class LayerScale(nn.Module):
    r"""
    Layer Scale from CaiT (Figure 1 (d)): https://arxiv.org/abs/2103.17239

    Note: We replace `lambda` used in the official paper with `alpha`
    Note: It only applies `pre_norm` on `x`. To normalize `other_inputs`, please either concatenate with `x` or add extrax normalization layers before `LayerScale`.
    """
    def __init__(
        self, 
        dim: int, 
        core_block: Union[nn.Module, str], 
        pre_norm: Union[nn.Module, str] = "LayerNorm", 
        alpha: float = 1e-4, 
        path_dropout: float = 0., 
        **kwargs  # kwargs for the `core_block`
    ) -> None:
        super().__init__()

        self.pre_norm = pre_norm(dim) if not isinstance(pre_norm, str) and issubclass(pre_norm, nn.Module) else getattr(nn, pre_norm)(dim)
        self.aff_transform = torch.jit.script_if_tracing(AffineTransform(dim, alpha, beta=None))
        self.core_block = core_block(dim, **kwargs) if not isinstance(core_block, str) and issubclass(core_block, nn.Module) else getattr(nn, core_block)(dim, **kwargs)
        self.path_dropout = PathDropout(path_dropout)

    def forward(self, x: torch.Tensor, *other_inputs) -> torch.Tensor:
        transformed_x = self.core_block(self.pre_norm(x), *other_inputs)

        return x + self.path_dropout(self.aff_transform(transformed_x))
        