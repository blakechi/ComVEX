import torch
from torch import nn

class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()

        assert fn is not None, "[Residual] Must give it a function (normaly, a neural net)"
        self._fn = fn

    def forward(self, x, *args, **kwargs):
        return self._fn(x, *args, **kwargs) + x


class Norm(nn.Module):
    def __init__(self, fn=None, *, dim):
        super(Norm, self).__init__()
        self._fn = fn
        self._norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        if self._fn:
            return self._fn(self._norm(x), *args, **kwargs)

        return self._norm(x)

class FeedForward(nn.Module):
    def __init__(self, *, dim=None, hidden_dim=None, residual=False, dropout=0.0):
        super().__init__()
        assert dim is not None, "[FeedForward] Must specify the in/out dim"
        assert hidden_dim is not None, "[FeedForward] Must specify the hidden dim"

        if residual:
            self._net = nn.Sequential(
                Residual(
                    nn.Sequential(
                        nn.Linear(dim, hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    )
                ),
                nn.Linear(hidden_dim, dim),
            )
        else:
            self._net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
            )

    def forward(self, x):
        return self._net(x)