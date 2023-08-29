import torch

def matrix_power(matrix: torch.Tensor, power: int):
    result = torch.eye(matrix.size(0), device=matrix.device, dtype=matrix.dtype)
    for _ in range(power):
        result = torch.mm(result, matrix)
    return result