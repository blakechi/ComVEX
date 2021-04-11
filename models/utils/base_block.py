import torch
from torch import nn

class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()

        assert fn is not None, f"[{self.__class__.__name__}] Must give it a function (normaly, a neural net)"
        self._fn = fn

    def forward(self, x, *args, **kwargs):
        return self._fn(x, *args, **kwargs) + x


class LayerNorm(nn.Module):
    def __init__(self, fn=None, *, dim=None, use_pre_norm=False):
        super(LayerNorm, self).__init__()
        self._fn = fn
        self._norm = nn.LayerNorm(dim)
        self._use_pre_norm = use_pre_norm

    def forward(self, x, *args, **kwargs):
        if self._fn:
            if self._use_pre_norm:
                return self._fn(self._norm(x), *args, **kwargs)
            else:
                return self._norm(self._fn(x, *args, **kwargs))

        return self._norm(x)


class MaskLayerNorm(LayerNorm):
    """
    Args:
        x (b, n, d): input tensor
        norm_mask (b, n) (Bool Tensor): True => to 0, False => ignore 
    """

    def forward(self, x, norm_mask=None, *args, **kwargs):
        if self._fn:
            if self._use_pre_norm:
                x = self._fn(self._norm(x), *args, **kwargs)
            else:
                x = self._norm(self._fn(x, *args, **kwargs))
        else:
            x = self._norm(x)

        assert norm_mask is not None, f"[{self.__class__.__name__}] Please provide `norm_mask`."

        return x.masked_fill_(norm_mask, 0)


class FeedForward(nn.Module):
    def __init__(self, *, dim=None, hidden_dim=None, output_dim=None, ff_dropout=0.0, useNorm=False):
        super().__init__()
        assert dim is not None, f"[{self.__class__.__name__}] Must specify the input dim"
        assert hidden_dim is not None, f"[{self.__class__.__name__}] Must specify the hidden dim"

        out_dim = output_dim if output_dim is not None else dim

        if useNorm:
            self._net = nn.Sequential(
                LayerNorm(
                    nn.Sequential(
                        nn.Linear(dim, hidden_dim),
                        nn.Dropout(ff_dropout),
                    ),
                    dim=dim
                ),
                nn.GELU(),
                LayerNorm(
                    nn.Linear(hidden_dim, out_dim),
                    dim=hidden_dim
                ),
            )
        else:
            self._net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(ff_dropout),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, x):
        return self._net(x)