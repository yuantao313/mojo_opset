"""Matrix multiplication with exact provider dispatch."""

from typing import Optional

import torch

from ._checks import _require_inference
from ._dispatch import load_impl


def mm(a: torch.Tensor, b: torch.Tensor, *, implementation: Optional[str] = None) -> torch.Tensor:
    """Return the matrix product of two rank-2 tensors; forward-only."""

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"mm requires rank-2 inputs, got {a.ndim} and {b.ndim}.")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"mm requires a.shape[1] == b.shape[0], got {tuple(a.shape)} and {tuple(b.shape)}.")
    if a.dtype != b.dtype:
        raise ValueError(f"mm requires matching dtypes, got {a.dtype} and {b.dtype}.")
    if a.device != b.device:
        raise ValueError(f"mm requires matching devices, got {a.device} and {b.device}.")
    _require_inference("mm", a, b)
    forward, _ = load_impl("mm", implementation)
    return forward(a, b)


def matmul(a: torch.Tensor, b: torch.Tensor, *, implementation: Optional[str] = None) -> torch.Tensor:
    """Return the batched matrix product with ``torch.matmul`` broadcasting; forward-only."""

    if a.ndim < 1 or b.ndim < 1:
        raise ValueError(f"matmul requires rank >= 1 inputs, got {a.ndim} and {b.ndim}.")
    if a.shape[-1] != b.shape[-2 if b.ndim > 1 else 0]:
        raise ValueError(f"matmul requires matching contracted dimensions, got {tuple(a.shape)} and {tuple(b.shape)}.")
    if a.dtype != b.dtype:
        raise ValueError(f"matmul requires matching dtypes, got {a.dtype} and {b.dtype}.")
    if a.device != b.device:
        raise ValueError(f"matmul requires matching devices, got {a.device} and {b.device}.")
    _require_inference("matmul", a, b)
    forward, _ = load_impl("matmul", implementation)
    return forward(a, b)
