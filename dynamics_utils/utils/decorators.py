from functools import wraps

import numpy as np
import torch

def to_tensor(data):
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data), 'numpy'
    elif isinstance(data, torch.Tensor):
        return data, 'torch'
    return data, None

def to_original_type(data, original_type):
    if original_type == 'numpy':
        return data.numpy()
    return data

def to_original_types(data, original_types):
    if isinstance(data, tuple):
        return tuple(to_original_type(d, t) for d, t in zip(data, original_types))
    else:
        return to_original_type(data, original_types[0])

def ensure_tensor(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Convert input arguments to tensors
        converted_args = []
        original_types = []
        for arg in args:
            tensor_arg, original_type = to_tensor(arg)
            converted_args.append(tensor_arg)
            original_types.append(original_type)

        converted_kwargs = {}
        for key, value in kwargs.items():
            tensor_value, _ = to_tensor(value)
            converted_kwargs[key] = tensor_value

        # Execute the function
        result = func(*converted_args, **converted_kwargs)

        # Convert result(s) back to the original type
        result = to_original_types(result, original_types)

        return result

    return wrapper
