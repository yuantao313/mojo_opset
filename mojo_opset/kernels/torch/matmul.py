import torch


def mm_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mm(a, b)


def matmul_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b)
