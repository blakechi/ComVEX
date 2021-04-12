import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from models.utils import Residual, LayerNorm, FeedForward, MultiheadAttention


class MAB(nn.Module):
    def __init__(
        self, *, dim, heads, ff_dim_scale=4, pre_norm=False, **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads

        self.attention = LayerNorm(
            dim=dim,
            cross_dim=dim,
            use_pre_norm=pre_norm,
            use_cross_attention=True,
            fn=Residual(
                fn=MultiheadAttention(
                    dim=dim,
                    heads=self.heads, 
                    **kwargs
                ),
            ),
        )
        self.ff = LayerNorm(
            dim=dim,
            use_pre_norm=pre_norm,
            fn=Residual(
                fn=FeedForward(
                    dim=dim, 
                    hidden_dim=ff_dim_scale*dim, 
                    **kwargs
                )
            ),
        )

    def forward(self, query, key_value, attention_mask=None):
        out = self.attention((query, key_value, key_value), attention_mask=attention_mask)

        return self.ff(out)


class SAB(nn.Module):
    def __init__(
        self, *, dim, heads, ff_dim_scale=4, pre_norm=False, **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads

        self.attention = LayerNorm(
            dim=dim,
            use_pre_norm=pre_norm,
            fn=Residual(
                fn=MultiheadAttention(
                    dim=dim,
                    heads=self.heads, 
                    **kwargs
                ),
            ),
        )
        self.ff = LayerNorm(
            dim=dim,
            use_pre_norm=pre_norm,
            fn=Residual(
                fn=FeedForward(
                    dim=dim, 
                    hidden_dim=ff_dim_scale*dim, 
                    **kwargs
                )
            ),
        )

    def forward(self, x, attention_mask=None):
        out = self.attention(x, attention_mask=attention_mask)

        return self.ff(out)


class ISAB(nn.Module):
    def __init__(
        self, 
        *, 
        dim, 
        heads, 
        num_inducing_points, 
        attention_dropout=0.0, 
        ff_dropout=0.0, 
        ff_dim_scale=4, 
        pre_norm=False, 
        head_dim=None
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.num_inducing_points = num_inducing_points
        assert (
            self.num_inducing_points > 0
        ), "Number of inducing points must be greater than zero."

        self.inducing_points = torch.nn.Parameter(
            torch.zeros(self.num_inducing_points, self.dim)
        )
        torch.nn.init.kaiming_normal_(self.inducing_points)

        self.first_MAB = MAB(
            dim=self.dim,
            heads=self.heads,
            head_dim=head_dim,
            attention_dropout=attention_dropout,
            ff_dropout=ff_dropout,
            ff_dim_scale=ff_dim_scale,
            pre_norm=pre_norm
        )
        self.second_MAB = MAB(
            dim=self.dim,
            heads=self.heads,
            head_dim=head_dim,
            attention_dropout=attention_dropout,
            ff_dropout=ff_dropout,
            ff_dim_scale=ff_dim_scale,
            pre_norm=pre_norm
        )

    def forward(self, x, attention_mask=None):
        b, n, e = x.shape
        inducing_points = repeat(self.inducing_points, "i e -> b i e", b=b)

        out = self.first_MAB(inducing_points, x, attention_mask)  
        attention_mask = attention_mask.transpose(-1, -2)
        out = self.second_MAB(x, out, attention_mask)

        return out


class SetTransformer(nn.Module):
    def __init__(self):
        ...
    
    def forward(self, x, attention_mask):
        ...