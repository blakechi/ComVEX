import torch
import torch.nn.functional as F
from torch import nn

from models.utils import Residual, LayerNorm, FeedForward


class FNetFourierTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.fft.fft(x, dim=-1)
        x = torch.fft.fft(x, dim=-2)

        return x.real


class FNetEncoderLayer(nn.Module):
    def __init__(self, *, dim, pre_norm=False, ff_dim=None, **kwargs):
        super().__init__()

        self.fourier_block = LayerNorm(
            Residual(
                FNetFourierTransform()
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

    def forward(self, x):
        x = self.fourier_block(x)

        return self.ff_block(x)


class FNetBackbone(nn.Module):
    def __init__(
        self, 
        *, 
        dim, 
        depth, 
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.layers = nn.Sequential(*[
            FNetEncoderLayer(
                dim=self.dim,  
                **kwargs
            ) for _ in range(depth)
        ])
                
    def forward(self, x):
        return self.layers(x)


class FNetWithLinearClassifier(nn.Module):
    def __init__(self, config=None) -> None:
        super().__init__()

        # self.token_embedding
        self.fnet_backbone = FNetBackbone(**config.__dict__)
        self.classifier = nn.LazyLinear(config.num_classes)

    def forward(self, x):
        x = self.fnet_backbone(x)

        return self.classifier(x)



