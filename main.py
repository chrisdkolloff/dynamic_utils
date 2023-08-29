#%%
import numpy as np
import torch
from dynamics_utils.utils.decorators import ensure_tensor

a = np.array([1, 2, 3])
b = torch.tensor([1, 2, 3])

from dynamics_utils.msm import calculate_acf

calculate_acf(a, a)