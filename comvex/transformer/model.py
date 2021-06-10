from torch import nn
from einops import repeat

from comvex.utils import Residual, LayerNorm, FeedForward, MultiheadAttention


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, heads=None, head_dim=None, pre_norm=False, ff_dim=None, **kwargs):
        super().__init__()

        self.attention_block = LayerNorm(
            Residual(
                MultiheadAttention(
                    dim, heads=heads, head_dim=head_dim, **kwargs
                )
            ),
            dim=dim,
            use_pre_norm=pre_norm
        )
        self.ff_block = LayerNorm(
            Residual(
                FeedForward(
                    dim=dim, hidden_dim=ff_dim if ff_dim is not None else 4*dim, **kwargs
                )
            ),
            dim=dim,
            use_pre_norm=pre_norm
        )

    def forward(self, x, attention_mask=None):
        x = self.attention_block(x, attention_mask)

        return self.ff_block(x)


class Transformer(nn.Module):
    def __init__(
        self, *, dim, heads, depth=12, **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                dim=self.dim, 
                heads=self.heads,  
                **kwargs
            )
            for _ in range(depth)
        ])
                
    def make_attention_mask(self, padding_mask):
        """
        padding_mask:

        True:  mask
        False: ignore
        """

        return repeat(padding_mask[:, :, None] + padding_mask[:, None, :], "b n m -> b h n m", h=h)
        
    def forward(self, x, attention_mask=None, padding_mask=None):
        if padding_mask is not None:
            attention_mask |= self.make_attention_mask(padding_mask)

        for layer in self.layers:
            x = layer(x, attention_mask)

        return x