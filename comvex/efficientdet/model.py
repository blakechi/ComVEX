from functools import partial
from collections import OrderedDict, namedtuple
from os import name
from typing import Optional, Union, Dict, NamedTuple

import torch
from torch import nn
try:
    from typing_extensions import Final
except:
    from torch.jit import Final

from comvex.utils import EfficientNetBase, MBConvXd
from comvex.utils.helpers import get_attr_if_exists, config_pop_argument
