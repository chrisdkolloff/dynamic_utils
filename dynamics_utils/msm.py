from typing import Union

import numpy as np
import torch

from dynamics_utils.utils.decorators import ensure_tensor

@ensure_tensor
def timescales_from_eigvals(eigvals: Union[np.ndarray, torch.Tensor], lag: int = 1, dt_traj: float = 1.) \
        -> Union[np.ndarray, torch.Tensor]:
    """
    Calculate timescales from eigenvalues
    Parameters
    ----------
    eigvals :    Union[np.ndarray, torch.Tensor], shape (n, )
        eigenvalues
    lag :        int, default=1
        lag time
    dt_traj :    float, default=1.
        trajectory timestep
    Returns
    -------
    Union[np.ndarray, torch.Tensor], shape (n, )

    """
    return - dt_traj * lag / torch.log(torch.abs(eigvals))

@ensure_tensor
def amplitudes_from_observables(a: Union[np.ndarray, torch.Tensor], leigvecs: Union[np.ndarray, torch.Tensor]) \
        -> Union[np.ndarray, torch.Tensor]:
    """
    Calculate amplitudes from observables
    Parameters
    ----------
    a :          Union[np.ndarray, torch.Tensor], shape (n, )
        average observable per state
    leigvecs :   Union[np.ndarray, torch.Tensor], shape (n, n)
        left eigenvectors
    Returns
    -------
    Union[np.ndarray, torch.Tensor], shape (n, )
    """
    return a.matmul(leigvecs) ** 2