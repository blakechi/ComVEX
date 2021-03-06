import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from models.utils import Residual, Norm, FeedForward, MultiheadAttention


class Transformer(nn.Module):
    def __init__(
        self, *, dim, heads=4, depth, head_dim=None, ff_dim, ff_dropout=0.0, max_seq_len=128
    ):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.max_seq_len = max_seq_len

        self.layers = nn.ModuleList([
            nn.ModuleList(
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