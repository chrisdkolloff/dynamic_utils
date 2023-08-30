from typing import Union, List, Any

import torch
import numpy as np

from .decorators import ensure_tensor


def matrix_power(matrix: torch.Tensor, power: int):
    result = torch.eye(matrix.size(0), device=matrix.device, dtype=matrix.dtype)
    for _ in range(power):
        result = torch.mm(result, matrix)
    return result


@ensure_tensor
def pad_and_stack(list_of_tensors: List[Union[np.ndarray, torch.Tensor]],
                  axis: int = 1,
                  value: Any = torch.nan) \
        -> torch.Tensor:
    """
    Pad tensors of different lengths and stack them along a given axis.
    Expected shape per array: (n_obs, n_datapoints)

    Parameters
    ----------
    list_of_tensors:    Union[np.ndarray, torch.Tensor]
        list of tensors to pack and stack

    axis:   int
        axis along which to stack and pad

    value:  Any, default: torch.nan
        value with which to pad tensors

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        padded and stacked tensors

    """

    # make sure list_of_tensors is at least 2d
    list_of_tensors = [torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr for arr in list_of_tensors ]
    list_of_tensors = [torch.atleast_2d(arr) for arr in list_of_tensors]

    padded_tensors = []
    max_len = max([arr.shape[axis] for arr in list_of_tensors])

    for arr in list_of_tensors:
        len_arr = arr.shape[1]
        parr = torch.nn.functional.pad(arr, (0, max_len - len_arr), value=value)
        padded_tensors.append(parr)
    if axis == 0:
        return torch.hstack(padded_tensors)
    elif axis == 1:
        return torch.vstack(padded_tensors)
    else:
        raise ValueError(f'Stacking only works for axis 0 or 1, not axis = {axis}')


@ensure_tensor
def masked_mean(tensor: Union[np.ndarray, torch.Tensor],
                mask: Union[np.ndarray, torch.Tensor],
                axis: int = 1)\
        -> Union[np.ndarray, torch.Tensor]:
    """
    Calculates the mean along a certain axis of a masked tensor

    Parameters
    ----------
    tensor:     Union[np.ndarray, torch.Tensor]
        Tensor whose mean should be calculated

    mask:   Union[np.ndarray, torch.Tensor]
        Boolean mask, same shape as tensor

    axis:   int, default: 1
        Axis along which to calculate mean

    Returns
    -------
    masked_mean:    Union[np.ndarray, torch.Tensor]
        Mean of masked tensor
    """
    tensor[~mask] = 0.0
    masked = torch.mul(tensor, mask)
    return masked.sum(dim=axis) / mask.sum(dim=axis)