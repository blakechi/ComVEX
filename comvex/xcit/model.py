from typing import Optional, Tuple
from collections import OrderedDict

import torch
from torch import nn
from einops import repeat

from comvex.vit import ViTBase
from comvex.utils import MLP, TokenDropout, ProjectionHead
from comvex.utils.helpers.functions import config_pop_argument
from .config import XCiConfig

