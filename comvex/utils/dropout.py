from typing import Optional
import torch
from torch import nn


def dimension_wise_dropout(x: torch.Tensor, dropout_rate: float = .0, dim: int = -1, training: bool = False) -> torch.Tensor:
    r"""
    Drop the specified dimension of the input tensor `x`.
    Reference from: https://github.com/rwightman/pytorch-image-models/blob/54a6cca27a9a3e092a07457f5d56709da56e3cf5/timm/models/layers/drop.py#L140

    Example:
    b, n, d = x.shape

    # Path dropout
    x = x + dimension_wise_dropout(your_module(x), 0.2, 0, training=True)

    # Batch-wise dropout
    x = dimension_wise_dropout(x, 0.2, 0, training=True)

    # Token-wise dropout
    x = dimension_wise_dropout(x, 0.2, -2, training=True)

    # Feature-wise dropout
    x = dimension_wise_dropout(x, 0.2, -1, training=True)
    """

    if dropout_rate == 0. or not training: 
        return x

    assert (
        0. < dropout_rate and dropout_rate < 1.
    ), f"[dimension_wise_dropout] Dropout rate should be greater than 0 and less than 1, but got {dropout_rate}"

    keep_rate = 1. - dropout_rate
    shape = (1,)*x.ndim
    shape[dim] = x.shape[dim]
    mask = (torch.rand(shape, dtype=x.dtype, device=x.device) + keep_rate).floor_()

    return x*(1./keep_rate)*mask


class TokenWiseDropout(nn.Module):
    r"""
    Dropout per tokens
    """
    def __init__(self, p: float = 0.) -> None:  
        super().__init__()
        self.dropout_rate = p
        self.scale = 1./(1. - p)
        self.bernoulli = torch.distributions.bernoulli.Bernoulli(probs=1. - p)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.BoolTensor] = None, preserve_tokens: int = 0) -> torch.Tensor:
        r"""
        Args:
            x (b, n, d): The inpute tensor.
            padding_mask (b, n): True means to mask and False means to drop. Default: None
            preserve_tokens: When a sequence's or data sample's number of tokens is less than this number,
                its tokens won't be dropped. Need to be smaller than `n`. Default: 0
        """
        b, n, _ = x.shape

        if self.dropout_rate == 0. or not self.training:
            return x

        mask = self.bernoulli.sample((b, n))
        if padding_mask is not None:
            # token-wise dropout | preserve padding tokens | preserve sequences that are too short
            mask = mask.to(torch.bool) | padding_mask | (padding_mask.sum(dim=-1) > (n - preserve_tokens)).unsqueeze(-1)
            mask = mask.to(x.dtype)

        return x*self.scale*(mask.unsqueeze(-1))


class TokenDropout(TokenWiseDropout):
    r"""
    Dropout for token features supported with padding masks
    """
    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.BoolTensor] = None, preserve_tokens: int = 0) -> torch.Tensor:
        r"""
        Args:
            x (b, n, d): The inpute tensor.
            padding_mask (b, n): True means to mask and False means to drop. Default: None
            preserve_tokens: When a sequence's or data sample's number of tokens is less than this number,
                its tokens won't be dropped. Need to be smaller than `n`. Default: 0
        """
        b, n, d = x.shape

        if self.dropout_rate == 0. or not self.training:
            return x

        mask = self.bernoulli.sample((b, n, d))
        if padding_mask is not None:

            mask = (
                mask.to(torch.bool) |  # toke features dropout
                padding_mask.unsqueeze(-1) |  # preserve padding tokens
                (padding_mask.sum(dim=-1) > (n - preserve_tokens))[(..., ) + (None,)*2]  # preserve sequences that are too short (unsqueeze at the last dimension twice)
            )
            mask = mask.to(x.dtype)

        return x*self.scale*mask


class PathDropout(nn.Module):
    def __init__(self, p: float = 0.) -> None:  
        super().__init__()
        self.dropout_rate = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x (b, ...): The inpute tensor. Assume `batch` is at dimension 0.
        """
        return dimension_wise_dropout(x, self.dropout_rate, 0, training=self.training)
