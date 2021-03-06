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
    def __init__(self, *, dim=None, hidden_dim=None, output_dim=None, dropout=0.0, useResidualWithNorm=False):
        super().__init__()
        assert dim is not None, "[FeedForward] Must specify the input dim"
        assert hidden_dim is not None, "[FeedForward] Must specify the hidden dim"

        out_dim = output_dim if output_dim is not None else dim

        if useResidualWithNorm:
            self._net = nn.Sequential(
                Residual(
                    Norm(
                        nn.Sequential(
                            nn.Linear(dim, hidden_dim),
                            nn.GELU(),
                            nn.Dropout(dropout),
                        ),
                        dim=dim
                    )
                ),
                Norm(
                    nn.Linear(hidden_dim, out_dim),
                    dim=hidden_dim
                ),
            )
        else:
            self._net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, x):
        return self._net(x)