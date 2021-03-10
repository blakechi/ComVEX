import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from models.utils import Residual, Norm, FeedForward, MultiheadAttention


class TransformerEncoder(nn.Module):
    def __init__(self, *, dim, heads, head_dim, ff_dim, ff_dropout):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                Residual(
                    Norm(
                        MultiheadAttention(
                            embedding_dim=dim, heads=heads, head_dim=head_dim
                        ),
                        dim=dim,
                    ),
                ),
                Residual(
                    Norm(
                        FeedForward(
                            dim=dim, hidden_dim=ff_dim, dropout=ff_dropout
                        ),
                        dim=dim,
                    ),
                ),
            ]
        )

    def forward(self, x):
        return self.layers(x)


class Transformer(nn.Module):
    def __init__(
        self, *, dim, heads, depth=12, head_dim=None, ff_dim=None, ff_dropout=0.0, max_seq_len=128
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.max_seq_len = max_seq_len

        self.head_dim = head_dim if head_dim is not None else embedding_dim // heads
        assert (
            self.head_dim * self.heads == self.embedding_dim
        ), "Head dimension times the number of heads must be equal to embedding dimension"

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                dim=self.dim, 
                heads=self.heads, 
                head_dim=self.head_dim, 
                ff_dim=ff_dim, 
                ff_dropout=ff_dropout
            )
            for _ in range(depth)
        ])
                
    def forward(self, x, att_mask=None, padding_mask=None):
        # device = x.device

        if padding_mask is not None:
            """
            att_mask / padding_mask:

            True: Ignore
            False: To mask
            """
            att_mask &= rearrange(
                repeat(
                    padding_mask[:, :, None], "b n 1 -> b n m", m=att_mask.shape[-1]
                ),
                "b n m -> b 1 n m",
            )

        for (attn, ff) in self.layers:
            x = attn(
                x,
                att_mask=att_mask,
            )
            x = ff(x)

        return x