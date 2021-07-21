from comvex.utils.helpers.functions import name_with_msg
from functools import partial
from collections import OrderedDict, namedtuple
from os import name
from typing import Literal, Optional, Union, List, Tuple, Dict,

import torch
from torch import nn
from torch.nn import functional as F
try:
    from typing_extensions import Final
except:
    from torch.jit import Final

from comvex.utils import EfficientNetBase, SeperateConvXd, XXXConvXdBase
from comvex.utils.helpers import get_attr_if_exists, config_pop_argument
from .bifpn import BiFPN


class EfficientDetDetector(nn.Module):
    def __init__(
        self,
        dim: int,
        backbone: nn.Module,
    ) -> None:
        super().__init__()