from typing import Union

import torch
import numpy as np

from .utils.decorators import ensure_tensor


@ensure_tensor
def random_float_between_interval(a: Union[int, float], b: Union[int, float], shape1: int = 1, shape2: int = 1)\
        -> torch.Tensor:
    """
    Generates random float on the interval [r1, r2]
    Parameters
    ----------
    a:     Union[int, float]
        lower bound
    b:     Union[int, float]
        upper bound
    shape1: int, default=1
        shape of first dimension
    shape2: int, default=1
        shape of second dimension

    Returns
    -------
    torch.Tensor, (shape1, shape2)

    """
    return (a - b) * torch.rand(shape1, shape2) + b


@ensure_tensor
def scale_to_range(arr: Union[np.ndarray, torch.Tensor], a: Union[float, int], b: Union[float, int], axis: int = 0)\
        -> Union[np.ndarray, torch.Tensor]:
    """
    Scales tensor to range between [a, b]
    Parameters
    ----------
    arr:    Union[np.ndarray, torch.Tensor]
        array/tensor to be scaled
    a:      Union[float, int]
        lower bound
    b:      Union[float, int]
        upper bound
    axis:  int, default=0
        axis to scale over

    Returns
    -------
    Union[np.ndarray, torch.Tensor]

    """
    min = torch.min(arr, dim=axis)[0]
    max = torch.max(arr, dim=axis)[0]
    return (b - a) * (arr - min) / (max - min) + a


