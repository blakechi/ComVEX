import torch
from torch import nn
from einops import repeat

from comvex.utils import Residual, LayerNorm, FeedForward, MultiheadAttention


class MAB(nn.Module):
    def __init__(
        self, *, dim, heads, ff_dim_scale=4, pre_norm=False, **kwargs
    ):
        super().__init__()

        self.attention = LayerNorm(
            dim=dim,
            cross_dim=dim,
            use_pre_norm=pre_norm,
            use_cross_attention=True,
            fn=Residual(
                fn=MultiheadAttention(
                    dim,
                    heads=heads, 
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

        self.attention = LayerNorm(
            dim=dim,
            use_pre_norm=pre_norm,
            fn=Residual(
                fn=MultiheadAttention(
                    dim,
                    heads=heads, 
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
        head_dim=None,
        **kwargs
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
        attention_mask = attention_mask.transpose(-1, -2) if attention_mask is not None else None
        out = self.second_MAB(x, out, attention_mask)

        return out


class PMA(nn.Module):
    def __init__(
        self, 
        *, 
        dim, 
        heads, 
        num_seeds=1, 
        attention_dropout=0.0, 
        ff_dim_scale=4, 
        pre_norm=False, 
        head_dim=None,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.num_seeds = num_seeds
        assert (
            self.num_seeds > 0
        ), "Number of seeds must be greater than zero."

        self.seeds = torch.nn.Parameter(
            torch.zeros(self.num_seeds, self.dim)
        )
        torch.nn.init.kaiming_normal_(self.seeds)

        self.MAB = MAB(
            dim=self.dim,
            heads=self.heads,
            head_dim=head_dim,
            attention_dropout=attention_dropout,
            ff_dim_scale=ff_dim_scale,
            pre_norm=pre_norm,
            **kwargs
        )
        self.ff = FeedForward(
            dim=dim, 
            hidden_dim=dim, 
            **kwargs
        )

    def forward(self, x, attention_mask=None):
        b, n, e = x.shape
        seeds = repeat(self.seeds, "s e -> b s e", b=b)

        return self.MAB(seeds, self.ff(x), attention_mask)


class SetTransformerEncoder(nn.Module):
    def __init__(self, internal_block=None, **kwargs):
        super().__init__()

        assert issubclass(internal_block, (SAB, ISAB)), f"[{self.__class__.__name__}] `internal_block` must be either `SAB` or `ISAB`."

        self.first_block = internal_block(**kwargs)
        self.second_block = internal_block(**kwargs)
    
    def forward(self, x, attention_mask):
        return self.second_block(self.first_block(x, attention_mask), attention_mask)


class SetTransformerDecoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.PMA = PMA(**kwargs)
        self.SAB = SAB(**kwargs)
        self.ff = FeedForward(**kwargs)
    
    def forward(self, x, attention_mask):
        return self.ff(self.SAB(self.PMA(x, attention_mask), attention_mask))


class SetTransformer(nn.Module):
    def __init__(
        self, 
        *, 
        dim, 
        heads, 
        encoder_base_block,
        num_inducing_points, 
        num_seeds, 
        attention_dropout=0.0, 
        ff_dropout=0.0, 
        ff_dim_scale=4, 
        pre_norm=False, 
        head_dim=None
    ):
        super().__init__()
    
        self.encoder = SetTransformerEncoder(
            dim=dim, 
            heads=heads, 
            internal_block=encoder_base_block,
            num_inducing_points=num_inducing_points, 
            attention_dropout=attention_dropout, 
            ff_dropout=ff_dropout, 
            ff_dim_scale=ff_dim_scale, 
            pre_norm=pre_norm, 
            head_dim=head_dim
        )
        self.decoder = SetTransformerDecoder(
            dim=dim, 
            heads=heads, 
            num_seeds=num_seeds, 
            attention_dropout=attention_dropout, 
            ff_dropout=ff_dropout, 
            ff_dim_scale=ff_dim_scale, 
            pre_norm=pre_norm, 
            head_dim=head_dim
        )
        self.output_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x, attention_mask=None):
        x = self.encoder(x, attention_mask=attention_mask)
        x = self.decoder(x, attention_mask=attention_mask)

        return self.output_head(x)