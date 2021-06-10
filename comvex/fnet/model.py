import torch
from torch import nn
from einops import repeat

from .config import FNetConfig
from comvex.vit import ViTBase
from comvex.utils import Residual, LayerNorm, FeedForward, ProjectionHead


class FNetFourierTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.fft.fft2(x).real


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
        dense_act_fnc_name="ReLU",
        **kwargs
    ):
        super().__init__()

        self.layers = nn.Sequential(*[
            FNetEncoderLayer(
                dim=dim,  
                **kwargs
            ) for _ in range(depth)
        ])
        self.dense = nn.Sequential(
            nn.Linear(dim, dim),
            getattr(nn, dense_act_fnc_name)(),
        )
        
    def forward(self, x):
        x = self.layers(x)

        return self.dense(x)


class FNetWithLinearClassifier(ViTBase):
    def __init__(self, config: FNetConfig = None) -> None:
        super().__init__(config.image_size, config.image_channel, config.patch_size)

        self.linear_proj = nn.Linear(self.patch_dim, config.dim, bias=False)
        self.CLS = nn.Parameter(torch.randn(1, 1, config.dim), requires_grad=True)
        self.position_code = nn.Parameter(torch.randn(1, self.num_patches + 1, config.dim))  # plus 1 for CLS
        self.token_dropout = nn.Dropout(config.token_dropout)

        self.backbone = FNetBackbone(**config.__dict__)

        self.proj_head = ProjectionHead(
            config.dim,
            config.num_classes,
            config.pred_act_fnc_name,
        )

    def forward(self, x):
        b, _, _, _ = x.shape  # b, c, h, w = x.shape

        x = self.patch_and_flat(x)
        x = self.linear_proj(x)
        x = self.token_dropout(x)

        # Prepend CLS token and add position code
        CLS = repeat(self.CLS, "1 1 d -> b 1 d", b=b)
        x = torch.cat([CLS, x], dim=1) + self.position_code

        x = self.backbone(x)

        return self.proj_head(x[:, 0, :])


