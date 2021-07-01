import torch
from torch import nn
from einops import repeat

from .config import FNetConfig
from comvex.vit import ViTBase
from comvex.utils import Residual, LayerNorm, FeedForward, ProjectionHead, TokenDropout
from comvex.utils.helpers import config_pop_argument


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
                    dim=dim, expand_dim=ff_dim if ff_dim is not None else 4*dim, **kwargs
                )
            ),
            dim=dim,
            use_pre_norm=pre_norm
        )

    def forward(self, x):
        x = self.fourier_block(x)

        return self.ff_block(x)


class FNetBackbone(ViTBase):
    def __init__(
        self, 
        *, 
        image_size,
        image_channel,
        patch_size,
        dim, 
        depth, 
        pre_norm,
        ff_dim,
        ff_dropout=0.0, 
        token_dropout=0.0, 
        ff_act_fnc_name="ReLU",
        dense_act_fnc_name="ReLU",
    ):
        super().__init__(image_size, image_channel, patch_size)

        self.linear_proj = nn.Linear(self.patch_dim, dim, bias=False)
        self.CLS = nn.Parameter(torch.randn(1, 1, dim), requires_grad=True)
        self.position_code = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))  # plus 1 for CLS
        self.token_dropout = TokenDropout(token_dropout)

        self.layers = nn.Sequential(*[
            FNetEncoderLayer(
                dim=dim,  
                pre_norm=pre_norm,
                f_dim=ff_dim,
                ff_dropout=ff_dropout,
                act_fnc_name=ff_act_fnc_name,
            ) for _ in range(depth)
        ])
        self.dense = nn.Sequential(
            nn.Linear(dim, dim),
            getattr(nn, dense_act_fnc_name)(),
        )
        
    def forward(self, x):
        b, _, _, _ = x.shape  # b, c, h, w = x.shape

        x = self.patch_and_flat(x)
        x = self.linear_proj(x)
        x = self.token_dropout(x)

        # Prepend CLS token and add position code
        CLS = repeat(self.CLS, "1 1 d -> b 1 d", b=b)
        x = torch.cat([CLS, x], dim=1) + self.position_code

        x = self.layers(x)

        return self.dense(x)


class FNetWithLinearClassifier(FNetBackbone):
    def __init__(self, config: FNetConfig = None) -> None:
        num_classes = config_pop_argument(config, "num_classes")
        pred_act_fnc_name = config_pop_argument(config, "pred_act_fnc_name")
        
        super().__init__(**config.__dict__)

        self.proj_head = ProjectionHead(
            config.dim,
            num_classes,
            pred_act_fnc_name,
        )

    def forward(self, x):
        x = super().forward(x)

        return self.proj_head(x[:, 0, :])


