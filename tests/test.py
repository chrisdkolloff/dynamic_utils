#%%
from dynamics_utils.msm import acf
import torch
from dynamics_utils.msm import amplitudes_from_observables

torch.manual_seed(5)

k = torch.rand(10)
a = torch.rand(20)
leigvecs = torch.rand(20, 20)
eigvals = torch.rand(20)[1:]
n_components = None
lagtime, dt_traj = 1, 1
stationary_distribution = leigvecs[:, 0]

def mean_center(arr, axis=1, keepdims=True):
    #  mean centers nd-array
    return arr - arr.mean(axis=axis, keepdims=keepdims)

a = mean_center(torch.atleast_2d(a))

amplitudes_m = torch.matmul(a, leigvecs[:, 1:])[:, :n_components]
amplitudes_o = amplitudes_from_observables(a, stationary_distribution)

amp_dyn = amplitudes_from_observables(a, leigvecs[:, 1:])[:, :n_components]

acf_m =(torch.matmul(a, stationary_distribution[:, None]) ** 2).T + \
          torch.matmul((eigvals[:n_components, None] ** (k * lagtime * dt_traj)).T, (amplitudes_m ** 2).T)

acf_o = amplitudes_o.T + torch.matmul((eigvals[:n_components, None] ** (k * lagtime * dt_traj)).T, (amplitudes_o).T)
acf = acf(k, a, leigvecs, eigvals)