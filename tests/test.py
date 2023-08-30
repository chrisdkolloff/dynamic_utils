#%%
import torch
from dynamics_utils.msm import calculate_stationary_observable

torch.manual_seed(5)

k = torch.rand(10)
a = torch.rand(20, 4)
leigvecs = torch.rand(20, 20)
eigvals = torch.rand(20)[1:]
n_components = None
lagtime, dt_traj = 1, 1
stationary_distribution = leigvecs[:, 0]
