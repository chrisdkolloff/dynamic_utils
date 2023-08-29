from typing import Union

import numpy as np
import torch

from dynamics_utils.utils.decorators import ensure_tensor
from dynamics_utils.msm import timescales_from_eigvals, amplitudes_from_observables


def test_function():
    print('Test passed!')

@ensure_tensor
def r1rho(nu1: Union[np.ndarray, torch.Tensor], c: Union[np.ndarray, torch.Tensor],
          tex: Union[np.ndarray, torch.Tensor], nu0: Union[float, int]) -> Union[np.ndarray, torch.Tensor]:
    """
    Calculate R1rho relaxation rate from chemical shift tensor and correlation time

    Parameters
    ----------
    nu1 :   Union[np.ndarray, torch.Tensor], shape (n, )
            Tensor of frequencies in Hz
    c :    Union[np.ndarray, torch.Tensor], shape(k, )
            Tensor of amplitudes
    tex :  Union[np.ndarray, torch.Tensor], shape(k, )
            Tensor of exchange timescales
    nu0:    Union[float, int]
            Magnetic field strength in MHz

    Returns
    -------
    Union[np.ndarray, torch.Tensor], shape(n, )
            Tensor of R1rho relaxation rates in Hz
    """
    return ((2 * torch.pi * nu0) ** 2 * (c[:, 1:].matmul(tex) / (1 + (tex * nu1[:, None]) ** 2)[:, :, None])).sum(
        axis=1).T

@ensure_tensor
def cpmg(nu1: Union[np.ndarray, torch.Tensor], c: Union[np.ndarray, torch.Tensor],
          tex: Union[np.ndarray, torch.Tensor], nu0: Union[float, int]) -> Union[np.ndarray, torch.Tensor]:
    """
    Calculate CPMG relaxation rate from chemical shift tensor and correlation time

    Parameters
    ----------
    nu1 :   Union[np.ndarray, torch.Tensor], shape(n, )
            Tensor of frequencies in Hz
    c :    Union[np.ndarray, torch.Tensor], shape(k, )
            Tensor of amplitudes
    tex :  Union[np.ndarray, torch.Tensor], shape(k, )
            Tensor of exchange timescales
    nu0:    Union[float, int]
            Magnetic field strength in MHz

    Returns
    -------
    Union[np.ndarray, torch.Tensor], shape(n, )
            Tensor of CPMG relaxation rates in Hz

    """
    tcp = 1 / (4 * nu1)
    return ((2 * torch.pi * nu0) ** 2 * (
            c[1:] * tex * (1 - (tex / tcp[:, None] * torch.tanh(tcp[:, None] / tex))))).sum(axis=1)

@ensure_tensor
def r1rho_msm(nu1, observable_by_state, leigvecs, eigvals, lag=1, dt_traj=1, nu0=None):
    tex = timescales_from_eigvals(eigvals, lag, dt_traj)
    c = amplitudes_from_observables(observable_by_state, leigvecs)
    return r1rho(nu1, c, tex, nu0)

@ensure_tensor
def cpmg_msm(nu1, observable_by_state, leigvecs, eigvals, lag=1, dt_traj=1, nu0=None):
    tex = timescales_from_eigvals(eigvals, lag, dt_traj)
    c = amplitudes_from_observables(observable_by_state, leigvecs)
    return cpmg(nu1, c, tex, nu0)


def ppm2Hz(cs: np.ndarray, nu0: Union[float, int]):
    """
    Convert array of chemical shifts in ppm to Hz

    Parameters
    ----------
    cs:     np.ndarray
            chemical shifts array in ppm
    nu0:    Union[float, int]
            Magnetic field strength in MHz

    Returns
    -------

    """
    return cs / 1e6 * nu0

def Hz2ppm(cs: np.ndarray, nu0: Union[float, int]):
    """
    Convert array of chemical shifts in Hz to ppm

    Parameters
    ----------
    cs:     np.ndarray
            chemical shifts array in Hz
    nu0:    Union[float, int]
            Magnetic field strength in MHz

    Returns
    -------

    """
    return cs / nu0 * 1e6


def calculate_larmor_frequency(proton_frequency: int, resonance='15N'):
    """

    Parameters
    ----------
    proton_frequency
    resonance

    Returns
    -------

    """
    gamma = {'15N': -27.117,  # in 1e6 rad s-1 T-1
             '1H': 267.522}

    magnetic_field_strength = {500: 11.7,  # in T
                               600: 14.1,
                               950: 22.3160}

    return - gamma[resonance] * magnetic_field_strength[proton_frequency] / (2 * torch.pi)
