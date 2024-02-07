from typing import Union, Tuple

import numpy as np
import torch
import deeptime
from deeptime.markov.tools.analysis import rdl_decomposition
from deeptime.markov import pcca

from .utils.decorators import ensure_tensor
from .utils.torch_utils import matrix_power
from .math import mean_center


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
                                        leigvecs: Union[np.ndarray, torch.Tensor]) \
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
                            dt_traj: float = 1.) \
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
    Union[np.ndarray, torch.Tensor]
        timescales
    Union[np.ndarray, torch.Tensor]
        amplitudes

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
    Union[np.ndarray, torch.Tensor]
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
    return torch.stack(
        [torch.mm(torch.mm(a * stationary_distribution, matrix_power(transition_matrix, ki)), a) for ki in k])


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
    if a.dim() == 1:
        a = a.unsqueeze(0)  # Convert from (n,) to (n, 1)
    a = mean_center(a)
    amplitudes_dynamic = amplitudes_from_observables(a, leigvecs[:, 1:])[:, :n_components]
    amplitudes_stationary = amplitudes_from_observables(a, stationary_distribution)
    acf = amplitudes_stationary.T + torch.matmul((eigvals[:n_components, None] ** (k * lag * dt_traj)).T,
                                                 (amplitudes_dynamic).T)
    return acf.T


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
    Union[np.ndarray, torch.Tensor]
        right eigenvectors
    Union[np.ndarray, torch.Tensor]
        eigenvalues
    Union[np.ndarray, torch.Tensor]
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
                      leigvecs: Union[np.ndarray, torch.Tensor]) \
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
    Union[np.ndarray, torch.Tensor] (n, n)
            transition matrix
    """
    return reigvecs.mm(eigvals).mm(leigvecs.T)


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
                       reigvecs: Union[np.ndarray, torch.Tensor]) \
        -> Union[np.ndarray, torch.Tensor]:
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
    Union[np.ndarray, torch.Tensor]
        left eigenvectors

    """
    return torch.diag(stationary_distribution).mm(reigvecs)


@ensure_tensor
def calculate_reigvecs(stationary_distribution: Union[np.ndarray, torch.Tensor],
                       orthonormal_reigvecs: Union[np.ndarray, torch.Tensor]) \
        -> Union[np.ndarray, torch.Tensor]:
    """
    Calculates the right eigenvectors from the orthonormal right eigenvectors and the stationary distribution
    Parameters
    ----------
    stationary_distribution:   Union[np.ndarray, torch.Tensor]
        stationary distribution
    orthonormal_reigvecs:   Union[np.ndarray, torch.Tensor]
        right eigenvectors

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        left eigenvectors

    """
    n = stationary_distribution.size()[0]
    reigvecs = orthonormal_reigvecs / torch.sqrt(stationary_distribution)[:, None]
    return torch.hstack([torch.ones(n, 1), reigvecs[:, 1:]])


@ensure_tensor
def calculate_stationary_observable(a: Union[np.ndarray, torch.Tensor],
                                    stationary_distribution: Union[np.ndarray, torch.Tensor]) \
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
    if a.dim() == 1:
        a = a.unsqueeze(-1)  # Convert from (n,) to (n, 1)
    return stationary_distribution.matmul(a)


@ensure_tensor
def calculate_average_observable_per_state(ftraj: Union[np.ndarray, torch.Tensor],
                                           dtraj: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Calculate average observable per state from feature trajectory

    Parameters
    ----------
    ftraj:  Union[np.ndarray, torch.Tensor]
        feature trajectory
    dtraj:  Union[np.ndarray, torch.Tensor]
        discrete trajectory

    Returns
    -------
    a:    Union[np.ndarray, torch.Tensor]
        mean value of feature trajectory in that particular state
    """
    assert len(ftraj) == len(dtraj)

    return torch.tensor([ftraj[dtraj == i].mean() for i in torch.unique(dtraj)])


@ensure_tensor
def calculate_free_energy_potential(stationary_distribution: Union[np.ndarray, torch.Tensor], kT: float = 1.0):
    """
    Calculates the free energy potential as FEP = - log(pi)

    Parameters
    ----------
    stationary_distribution:    Union[np.ndarray, torch.Tensor]
        stationary distribution
    kT:     float, default = 1.0
        Boltzmann factor

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        free energy potential
    """
    return - torch.log(stationary_distribution) * kT


@ensure_tensor
def calculate_metastable_decomposition(transition_matrix: Union[np.ndarray, torch.Tensor],
                                       n_metastable_states: int) \
        -> pcca:
    """
    Calculates the metastable decomposition of the transition matrix

    Parameters
    ----------
    transition_matrix:  Union[np.ndarray, torch.Tensor]
        transition matrix
    n_metastable_states:    int
        number of metastable states

    Returns
    -------
    deeptime.markov.pcca object
            PCCA object
    """
    return pcca(transition_matrix.numpy(), n_metastable_states)


@ensure_tensor
def calculate_metastable_trajectory(pcca: pcca, dtraj: Union[np.ndarray, torch.Tensor]):
    """
    Calculates the metastable trajectory from the PCCA object and the discrete trajectory

    Parameters
    ----------
    pcca:   deeptime.markov.pcca object
        PCCA object
    dtraj:  Union[np.ndarray, torch.Tensor]
        discrete trajectory

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        metastable trajectory
    """
    return torch.tensor(pcca.assignments[dtraj])


@ensure_tensor
def calculate_mfpt(transition_matrix: Union[torch.Tensor, np.ndarray, deeptime.markov.msm.MarkovStateModel],
                   pcca_assignments: Union[torch.Tensor, np.ndarray],
                   lag: float = 1,
                   dt_traj: float = 1.0) \
        -> Union[torch.Tensor, np.ndarray]:
    """
    Calculates the mean first passage time matrix

    Parameters
    ----------
    transition_matrix:  Union[torch.Tensor, np.ndarray, deeptime.markov.msm.MarkovStateModel]
        transition matrix
    pcca_assignments:   Union[torch.Tensor, np.ndarray]
        PCCA object
    lag:    float, default = 1
        lag time
    dt_traj:    float, default = 1.0
        trajectory timestep

    Returns
    -------
    Union[torch.Tensor, np.ndarray]
        Mean first passage time matrix
    """
    if isinstance(transition_matrix, torch.Tensor):
        transition_matrix = deeptime.markov.msm.MarkovStateModel(transition_matrix.numpy(), lagtime=lag * dt_traj)

    n_metastable_states = len(torch.unique(pcca_assignments))
    mfpt = torch.zeros((n_metastable_states, n_metastable_states))
    for i in range(n_metastable_states):
        for j in range(n_metastable_states):
            mfpt[i, j] = transition_matrix.mfpt(np.where(pcca_assignments == i)[0], np.where(pcca_assignments == j)[0])
    return mfpt


@ensure_tensor
def calculate_mfpt_rates(transition_matrix: Union[torch.Tensor, np.ndarray, deeptime.markov.msm.MarkovStateModel],
                         pcca_assignments: Union[torch.Tensor, np.ndarray],
                         lag: int = 1,
                         dt_traj: float = 1.0) \
        -> torch.Tensor:
    """
    Calculates the mean first passage time rates (inverse of MFPT)

    Parameters
    ----------
    transition_matrix:  Union[torch.Tensor, np.ndarray, deeptime.markov.msm.MarkovStateModel]
        transition matrix
    pcca_assignments:   Union[torch.Tensor, np.ndarray]
        PCCA object
    lag:    float, default = 1
        lag time
    dt_traj:    float, default = 1.0
        trajectory timestep

    Returns
    -------
    Union[torch.Tensor, np.ndarray]
        Mean first passage time rates
    """
    mfpt = calculate_mfpt(transition_matrix, pcca_assignments, lag, dt_traj)
    imfpt = torch.zeros_like(mfpt)
    a, b = torch.nonzero(mfpt).T
    imfpt[a, b] = 1 / mfpt[a, b]
    return imfpt


@ensure_tensor
def calculate_delta_G_1D(stationary_distribution: Union[torch.Tensor, np.ndarray],
                         barrier_state: int) \
        -> float:
    """
    Calculates the free energy difference between two states in a 1D system
    Parameters
    ----------
    stationary_distribution:    Union[torch.Tensor, np.ndarray]
        stationary distribution
    barrier_state:  int
        index of barrier state

    Returns
    -------
    float
    """
    return - torch.log(stationary_distribution[:barrier_state].sum() / stationary_distribution[barrier_state:].sum())


@ensure_tensor
def divide_trajectories(trajectories: list[torch.Tensor], thresholds: list[float]) -> list[list[torch.Tensor]]:
    """
    Divides a list of trajectories into subtrajectories based on threshold values.

    Parameters
    ----------
    trajectories:  List[torch.Tensor]
        List of trajectories, each trajectory is a tensor.
    thresholds: List[float]
        List of threshold values for dividing trajectories.

    Returns
    -------
    List[torch.Tensor]
        List of lists containing subtrajectories.
    """
    if not isinstance(trajectories, list):
        trajectories = [trajectories]

    if not isinstance(thresholds, list):
        thresholds = [thresholds]

    # Make sure all trajectories are 2D tensors
    trajectories = [torch.atleast_2d(traj) for traj in trajectories]

    # If not all trajectories have the same dimension, raise an error
    if len(set([traj.shape[1] for traj in trajectories])) > 1:
        raise ValueError("Not all trajectories have the same dimension.")

    # If number of thresholds is not equal to the dimension of the trajectories, broadcast thresholds
    if len(thresholds) != trajectories[0].shape[1]:
        thresholds = thresholds * trajectories[0].shape[1]

    divided_trajectories = []

    for traj in trajectories:
        subtrajectories = []
        start_idx = 0

        # Convert thresholds to a tensor for efficient broadcasting
        thresholds_tensor = torch.tensor(thresholds).view(1, -1)

        # Check for threshold crossings
        crosses_threshold = ((traj[:-1] < thresholds_tensor) & (traj[1:] >= thresholds_tensor)) | \
                            ((traj[:-1] >= thresholds_tensor) & (traj[1:] < thresholds_tensor))
        crossing_indices = torch.where(crosses_threshold.any(dim=1))[0] + 1

        # Split the trajectory based on crossing indices
        for idx in crossing_indices:
            subtrajectories.append(traj[start_idx:idx])
            start_idx = idx
        subtrajectories.append(traj[start_idx:])

        divided_trajectories.append(subtrajectories)

    return divided_trajectories
