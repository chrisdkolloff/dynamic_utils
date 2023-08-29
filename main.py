#%%
import numpy as np
import torch
from dynamics_utils.utils.decorators import ensure_tensor

a = np.array([1, 2, 3])
b = torch.tensor([1, 2, 3])

@ensure_tensor
def f(a, b):
    x = a.dot(b)
    return x

o = f(a, a)