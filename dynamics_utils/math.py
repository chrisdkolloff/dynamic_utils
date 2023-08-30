from typing import Union

import torch
import numpy as np

from .utils.decorators import ensure_tensor


@ensure_tensor
def random_float_between_interval(r1: Union[int, float], r2: Union[int, float], shape1: int = 1, shape2: int = 1)\
        -> Union[np.ndarray, torch.Tensor]:
    """
    Generates random float on the interval [r1, r2]
    Parameters
    ----------
    r1:     Union[int, float]
        lower bound
    r2:     Union[int, float]
        upper bound
    shape1: int, default=1
        shape of first dimension
    shape2: int, default=1
        shape of second dimension

    Returns
    -------

    """
    return (r1 - r2) * torch.rand(shape1, shape2) + r2

