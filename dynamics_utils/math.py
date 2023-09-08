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


@ensure_tensor
def normalize_in_range(vec: Union[np.ndarray, torch.Tensor],
                       a: Union[int, float] = -1,
                       b: Union[int, float] = 1,
                       axis: int = 1)\
        -> Union[np.ndarray, torch.Tensor]:
    """
    Normalizes vector in range [a, b]
    Parameters
    ----------
    vec:    Union[np.ndarray, torch.Tensor]
        vector
    a:      Union[int, float], default=-1
        lower bound
    b:      Union[int, float], default=1
        upper bound
    axis:   int, default=1
        axis to normalize over

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
    """
    max = torch.max(vec, dim=axis)[0]
    min = torch.min(vec, dim=axis)[0]
    return (b - a) * ((vec - min) / (max - min)) + a


@ensure_tensor
def is_stochastic(transition_matrix: Union[np.ndarray, torch.Tensor],
                  dtype: torch.dtype = torch.float64)\
        -> bool:
    """
    Checks if matrix is stochastic
    Parameters
    ----------
    transition_matrix:  Union[np.ndarray, torch.Tensor]
        transition matrix
    dtype:  torch.dtype, default=torch.float64
        dtype

    Returns
    -------
    bool

    """
    return torch.allclose(transition_matrix.sum(axis=1), torch.ones(transition_matrix.size()[0], dtype=dtype))


@ensure_tensor
def is_valid(transition_matrix: Union[np.ndarray, torch.Tensor]):
    """
    Checks if matrix is valid
    Parameters
    ----------
    transition_matrix:  Union[np.ndarray, torch.Tensor]

    Returns
    -------
    bool
    """
    return torch.logical_not(torch.any(transition_matrix < 0))


@ensure_tensor
def is_reversible(transition_matrix: Union[np.ndarray, torch.Tensor],
                  stationary_distribution: Union[np.ndarray, torch.Tensor],
                  rtol: float = 1e-15)\
        -> bool:
    """
    Checks if matrix is reversible
    Parameters
    ----------
    transition_matrix:  Union[np.ndarray, torch.Tensor]
        transition matrix
    stationary_distribution:    Union[np.ndarray, torch.Tensor]
        stationary distribution
    rtol:   float, default=1e-15
        relative tolerance
    Returns
    -------
    bool

    """
    return torch.allclose(stationary_distribution[:, None] * transition_matrix,
                         (stationary_distribution[:, None] * transition_matrix).T,
                          rtol=rtol)


@ensure_tensor
def mean_center(arr, axis=1, keepdims=True):
    #  mean centers nd-array
    return arr - arr.mean(axis=axis, keepdims=keepdims)


@ensure_tensor
def is_orthonormal(tensor: Union[np.ndarray, torch.Tensor], rtol: float = 1e-05) -> bool:
    """
    Checks if matrix is orthonormal
    Parameters
    ----------
    tensor: Union[np.ndarray, torch.Tensor]
        matrix
    rtol:   float, default=1e-05
        relative tolerance

    Returns
    -------
    bool
    """
    return torch.allclose(tensor.mm(tensor.T), torch.eye(tensor.size()[0]), rtol=rtol)