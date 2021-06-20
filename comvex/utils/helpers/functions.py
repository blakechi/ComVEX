from typing import Optional, Any
import warnings

from torch import nn

from comvex.utils.config_base import ConfigBase


def name_with_msg(instance: nn.Module, msg: str) -> str:
    return f"[{instance.__class__.__name__}] {msg}"


def config_pop_argument(config: ConfigBase = ConfigBase(), argument: str = "") -> Any:
    try:
        return config.__dict__.pop(argument)
    except:
        if len(config) == 0:
            raise ValueError("Please specify a non-empty configuration")
        elif argument == "":
            raise KeyError("Please specify a `argument`")
        else:  # raise KeyError as usual
            config.__dict__.pop(argument)

        
def get_attr_if_exists(module: nn.Module, attr: str) -> Optional[nn.Module]:
    if attr and hasattr(module, attr):
        return getattr(module, attr)

    warnings.warn(f"Can't find {attr} in {module}")
    
    return None


