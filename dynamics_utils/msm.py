from typing import Union, Tuple

import numpy as np
import torch
from deeptime.markov.tools.analysis import rdl_decomposition

from .utils.decorators import ensure_tensor
from .utils.torch_utils import matrix_power


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


@ensure_tensor
def amplitudes_from_observables_general(a: Union[np.ndarray, torch.Tensor], b: Union[np.ndarray, torch.Tensor],
                                        reigvecs: Union[np.ndarray, torch.Tensor],
                                        leigvecs: Union[np.ndarray, torch.Tensor])\
        -> Union[np.ndarray, torch.Tensor]:
    """
    General method to calculate amplitudes from observables
    #TODO: use this method in convenience method `amplitudes_from_observables`
    Parameters
    ----------
    a:        Union[np.ndarray, torch.Tensor], shape (n, )
        average observable per state
    b:        Union[np.ndarray, torch.Tensor], shape (n, )
        average second observable per state (for cross-correlation)
    reigvecs: Union[np.ndarray, torch.Tensor], shape (n, n)
        right eigenvectors
    leigvecs: Union[np.ndarray, torch.Tensor], shape (n, n)
        left eigenvectors

    Returns
    -------
    Union[np.ndarray, torch.Tensor], shape (n, )
    """
    pi = leigvecs[:, 0]
    return (pi * a).matmul(reigvecs) * leigvecs.T.matmul(b)

@ensure_tensor
def fingerprint_correlation(reigvecs: Union[np.ndarray, torch.Tensor],
                            eigvals: Union[np.ndarray, torch.Tensor],
                            leigvecs: Union[np.ndarray, torch.Tensor],
                            a: Union[np.ndarray, torch.Tensor],
                            b: Union[np.ndarray, torch.Tensor, None] = None,
                            lag: int = 1,
                            dt_traj: float = 1.)\
        -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """
    Convenience function to calculate fingerprint correlation from eigenvalues and eigenvectors
    # TODO: remove reigvecs and use leigvecs instead
    Parameters
    ----------
    reigvecs:   Union[np.ndarray, torch.Tensor], shape (n, n)
        right eigenvectors
    eigvals:    Union[np.ndarray, torch.Tensor], shape (n, )
        eigenvalues
    leigvecs:   Union[np.ndarray, torch.Tensor], shape (n, n)
        left eigenvectors
    a:          Union[np.ndarray, torch.Tensor], shape (n, )
        average observable per state
    b:         Union[np.ndarray, torch.Tensor], shape (n, )
        average second observable per state (for cross-correlation)
    lag:       int, default=1
        lag time
    dt_traj:   float, default=1.
        trajectory timestep

    Returns
    -------

    """
    if b is None:
        b = a
    timescales = timescales_from_eigvals(eigvals, lag, dt_traj)
    amplitudes = amplitudes_from_observables_general(a, b, reigvecs, leigvecs)
    return timescales, amplitudes

@ensure_tensor
def calculate_acf_from_trajectory(traj: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Calculates the normalised ACF from a trajectory
    chrisdkolloff: renamed from `acf_trajectory` to `calculate_acf_from_trajectory`

    Parameters
    ----------
    traj:   Union[np.ndarray, torch.Tensor]
        trajectory

    Returns
    -------
    acf:    np.ndarray
        autocorrelation function

    """
    traj = torch.atleast_3d(traj)
    traj = traj - traj.mean()
    acf = torch.nn.functional.conv1d(traj, traj)[len(traj[:]):]
    return acf

@ensure_tensor
def calculate_acf_from_transition_matrix(k: Union[np.ndarray, torch.Tensor],
                                         a: Union[np.ndarray, torch.Tensor],
                                         transition_matrix: Union[np.ndarray, torch.Tensor],
                                         stationary_distribution: Union[np.ndarray, torch.Tensor]) \
        -> Union[np.ndarray, torch.Tensor]:
    """
    Calculates the ACF directly from the transition matrix
        E[o(tau)o(t+ktau)] = a * diag(P) * T**k * a
    Parameters
    ----------
    k:          Union[np.ndarray, torch.Tensor], shape (k, )
        lag times
    a:          Union[np.ndarray, torch.Tensor], shape (n, )
        average observable per state
    transition_matrix:  Union[np.ndarray, torch.Tensor], shape (n, n)
        transition matrix
    stationary_distribution:    Union[np.ndarray, torch.Tensor], shape (n, )
        stationary distribution

    Returns
    -------
    Union[np.ndarray, torch.Tensor], shape (k, )

    """
    return torch.stack([torch.mm(torch.mm(a * stationary_distribution, matrix_power(transition_matrix, ki)), a) for ki in k])

@ensure_tensor
def calculate_acf_from_spectral_components(k: Union[np.ndarray, torch.Tensor],
                                           a: Union[np.ndarray, torch.Tensor],
                                           leigvecs: Union[np.ndarray, torch.Tensor],
                                           eigvals: Union[np.ndarray, torch.Tensor],
                                           lag: int = 1,
                                           dt_traj: float = 1.,
                                           n_components: int = None) \
        -> Union[np.ndarray, torch.Tensor]:
    """
    Calculates the ACF from the spectral components
        E[o(tau)o(t+ktau)] = (a stationary_distribution) ** 2 + sum_{i=2}((lambda_i**k)(a l_i) **2)

    Parameters
    ----------
    k:          Union[np.ndarray, torch.Tensor], shape (k, )
        lag times
    a:          Union[np.ndarray, torch.Tensor], shape (n, )
        average observable per state
    leigvecs:   Union[np.ndarray, torch.Tensor], shape (n, n)
        left eigenvectors
    eigvals:    Union[np.ndarray, torch.Tensor], shape (n, )
        eigenvalues
    lag:    int, default=1
        lag time
    dt_traj:    float, default=1.
        trajectory timestep
    n_components:   int, default=None
        number of components to use

    Returns
    -------
    Union[np.ndarray, torch.Tensor], shape (k, )
    """
    stationary_distribution = leigvecs[:, 0]
    a = mean_center(torch.atleast_2d(a))
    amplitudes_dynamic = amplitudes_from_observables(a, leigvecs[:, 1:])[:, :n_components]
    amplitudes_stationary = amplitudes_from_observables(a, stationary_distribution)
    acf = amplitudes_stationary.T + torch.matmul((eigvals[:n_components, None] ** (k * lag * dt_traj)).T, (amplitudes_dynamic).T)
    return acf.T

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


def is_reversible(transition_matrix: Union[np.ndarray, torch.Tensor],
                  stationary_distribution: Union[np.ndarray, torch.Tensor])\
        -> bool:
    """
    Checks if matrix is reversible
    Parameters
    ----------
    transition_matrix:  Union[np.ndarray, torch.Tensor]
    stationary_distribution:    Union[np.ndarray, torch.Tensor]

    Returns
    -------
    bool

    """
    return torch.allclose(stationary_distribution[:, None] * transition_matrix, (stationary_distribution[:, None] * transition_matrix).T, rtol=1e-15)

@ensure_tensor
def eigendecomposition(transition_matrix: Union[np.ndarray, torch.Tensor],
                       renormalise: bool = False):
    """
    Calculates the eigendecomposition of the transition matrix
    Parameters
    ----------
    transition_matrix:  Union[np.ndarray, torch.Tensor]
        transition matrix
    renormalise:    bool, default=False
        renormalise transition matrix

    Returns
    -------
    reigvecs:   Union[np.ndarray, torch.Tensor]
        right eigenvectors
    eigvals:    Union[np.ndarray, torch.Tensor]
        eigenvalues
    stationary_distribution:    Union[np.ndarray, torch.Tensor]
        stationary distribution

    """
    if renormalise:
        transition_matrix = row_normalise(transition_matrix)
    r, d, l = rdl_decomposition(transition_matrix)
    eigvals = torch.diag(torch.from_numpy(d.real))[1:]
    eigvals = eigvals.to(torch.float64)
    leigvecs = torch.from_numpy(l.T)
    reigvecs = torch.from_numpy(r)
    stationary_distribution = leigvecs[:, 0]
    return reigvecs, eigvals, stationary_distribution


@ensure_tensor
def rdl_recomposition(reigvecs: Union[np.ndarray, torch.Tensor],
                      eigvals: Union[np.ndarray, torch.Tensor],
                      leigvecs: Union[np.ndarray, torch.Tensor])\
        -> Union[np.ndarray, torch.Tensor]:
    """
    Calculate T = reigvecs * diag(eigvals) * leigvecs.T through RDL recomposition

    Parameters
    ----------
    reigvecs:   Union[np.ndarray, torch.Tensor] (n, n)
            right eigenvectors
    eigvals:    Union[np.ndarray, torch.Tensor] (n,)
            eigenvalues
    leigvecs:   Union[np.ndarray, torch.Tensor] (n, n)
            left eigenvectors

    Returns
    -------
    transition_matrix:  Union[np.ndarray, torch.Tensor] (n, n)
            transition matrix
    """
    return reigvecs.mm(eigvals).mm(leigvecs.T)

@ensure_tensor
def mean_center(arr, axis=1, keepdims=True):
    #  mean centers nd-array
    return arr - arr.mean(axis=axis, keepdims=keepdims)

@ensure_tensor
def row_normalise(tensor: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Row normalises matrix
    Parameters
    ----------
    tensor:    Union[np.ndarray, torch.Tensor]
        tensor

    Returns
    -------
    Union[np.ndarray, torch.Tensor]

    """
    return tensor / tensor.sum(axis=1, keepdims=True)


@ensure_tensor
def calculate_leigvecs(stationary_distribution: Union[np.ndarray, torch.Tensor],
                       reigvecs: Union[np.ndarray, torch.Tensor]):
    """
    Calculates the left eigenvectors from the right eigenvectors and the stationary distribution
    Parameters
    ----------
    stationary_distribution:   Union[np.ndarray, torch.Tensor]
        stationary distribution
    reigvecs:   Union[np.ndarray, torch.Tensor]
        right eigenvectors

    Returns
    -------
    leigvecs:   Union[np.ndarray, torch.Tensor]
        left eigenvectors

    """
    return torch.diag(stationary_distribution).mm(reigvecs)

@ensure_tensor
def calculate_stationary_observable(a: Union[np.ndarray, torch.Tensor],
                                    stationary_distribution: Union[np.ndarray, torch.Tensor])\
        -> Union[np.ndarray, torch.Tensor]:
    """
    Calculates the stationary observable
    Parameters
    ----------
    a:  Union[np.ndarray, torch.Tensor] (n, k)
        observable
    stationary_distribution:   Union[np.ndarray, torch.Tensor] (n,)

    Returns
    -------
    Union[np.ndarray, torch.Tensor] (k,)

    """
    return stationary_distribution.matmul(torch.atleast_2d(a))


# def calculate_observable_by_state(ftraj: np.ndarray, dtraj: np.ndarray) -> np.ndarray:
#     """
#     Calculate observable_by_state vector from feature trajectory
#
#     Parameters
#     ----------
#     ftraj:  np.ndarray
#             Feature trajectory
#     dtraj:  np.ndarray
#             Discrete trajectory
#
#     Returns
#     -------
#     observable_by_state:    np.ndarray
#                             mean value of feature trajectory in that particular state
#     """
#     assert len(ftraj) == len(dtraj)
#
#     return np.array([ftraj[dtraj == i].mean() for i in np.unique(dtraj)])
#
#
# def random_between_interval(r1, r2, shape1=1, shape2=1):
#     return (r1 - r2) * torch.rand(shape1, shape2) + r2
#
#
# def scale_to_range(arr: torch.Tensor, a: Union[float, int], b: Union[float, int], axis=0):
#     """
#     Scales tensor to range between [a, b]
#     Parameters
#     ----------
#     arr
#     a
#     b
#
#     Returns
#     -------
#
#     """
#     min = torch.min(arr, dim=axis)[0]
#     max = torch.max(arr, dim=axis)[0]
#     return (b - a) * (arr - min) / (max - min) + a
#
#
# def calculate_free_energy_potential(stationary_distribution: torch.Tensor, kT: float = 1.0):
#     """
#     Calculates the free energy potential as FEP = - log(pi)
#
#     Parameters
#     ----------
#     stationary_distribution:    torch.Tensor
#                                 Stationary distribution
#     kT:                         float, default = 1.0
#                                 Boltzmann factor
#
#     Returns
#     -------
#     FEP:                        torch.Tensor
#                                 Free energy
#     """
#     return - torch.log(stationary_distribution) * kT
#
#
# def calculate_metastable_decomposition(transition_matrix: torch.Tensor, n_metastable_states: int) -> object:
#     """
#     Calculates the metastable decomposition of the transition matrix
#
#     Parameters
#     ----------
#     transition_matrix:          torch.Tensor
#                                 Transition matrix
#     n_metastable_states:        int
#                                 Number of metastable states
#
#     Returns
#     -------
#     pcca:                       deeptime.markov.pcca object
#                                 PCCA object
#     """
#     return pcca(transition_matrix.numpy(), n_metastable_states)
#
#
# def calculate_metastable_trajectory(pcca: pcca, dtraj: torch.Tensor):
#     """
#     Calculates the metastable trajectory from the PCCA object and the discrete trajectory
#
#     Parameters
#     ----------
#     pcca:           deeptime.markov.pcca object
#                     PCCA object
#     dtraj:          np.ndarray
#                     Discrete trajectory
#
#     Returns
#     -------
#     metastable_traj:    torch.Tensor
#                         Metastable trajectory
#     """
#     return torch.tensor(pcca.assignments[dtraj])
#
# def calculate_mfpt(transition_matrix: Union[torch.Tensor, deeptime.markov.msm.MarkovStateModel],
#                    pcca_assignments: torch.Tensor, n_metastable_states: int, lagtime: float = 1.0):
#     """
#     Calculates the mean first passage time matrix
#
#     Parameters
#     ----------
#     transition_matrix:      Union[torch.Tensor, deeptime.markov.msm.MarkovStateModel]
#                             Transition matrix
#     pcca_assignments:       torch.Tensor
#                             PCCA object
#     n_metastable_states:    int
#                             Number of metastable states
#     lagtime:                float, default = 1
#                             Lagtime
#
#     Returns
#     -------
#     mfpt:                   torch.Tensor
#                             Mean first passage time matrix
#     """
#     if isinstance(transition_matrix, torch.Tensor):
#         transition_matrix = deeptime.markov.msm.MarkovStateModel(transition_matrix.numpy(), lagtime=lagtime)
#
#     mfpt = torch.zeros((n_metastable_states, n_metastable_states))
#     for i in range(n_metastable_states):
#         for j in range(n_metastable_states):
#             mfpt[i, j] = transition_matrix.mfpt(np.where(pcca_assignments == i)[0], np.where(pcca_assignments == j)[0])
#     return mfpt
#
# def calculate_mfpt_rates(transition_matrix: Union[torch.Tensor, deeptime.markov.msm.MarkovStateModel],
#                          pcca_assignments: torch.Tensor, n_metastable_states: int, lagtime: float = 1.0):
#     """
#     Calculates the mean first passage time rates
#
#     Parameters
#     ----------
#     transition_matrix:      Union[torch.Tensor, deeptime.markov.msm.MarkovStateModel]
#                             Transition matrix
#     pcca_assignments:       torch.Tensor
#                             PCCA object
#     n_metastable_states:    int
#                             Number of metastable states
#     lagtime:                float, default = 1
#                             Lagtime
#
#     Returns
#     -------
#     mfpt_rates:             torch.Tensor
#                             Mean first passage time rates
#     """
#     mfpt = calculate_mfpt(transition_matrix, pcca_assignments, n_metastable_states, lagtime)
#     imfpt = torch.zeros_like(mfpt)
#     a, b = torch.nonzero(mfpt).T
#     imfpt[a, b] = 1 / mfpt[a, b]
#     return imfpt
#
#
# def calculate_delta_g(stationary_distribution: torch.Tensor, barrier_state: int):
#     return - torch.log(stationary_distribution[:barrier_state].sum() / stationary_distribution[barrier_state:].sum())