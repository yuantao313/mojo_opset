"""``matmul`` compiled from a batch-invariant CATLASS Python kernel."""

import torch

from ._host import SUPPORTED_DTYPES
from ._host import launch


def _normalize(a: torch.Tensor, b: torch.Tensor):
    a_was_1d = a.ndim == 1
    b_was_1d = b.ndim == 1
    if a_was_1d:
        a = a.unsqueeze(0)
    if b_was_1d:
        b = b.unsqueeze(-1)
    if a.ndim < 2 or b.ndim < 2:
        raise ValueError("matmul expects tensors of rank 1 to 6")
    if a.ndim > 6 or b.ndim > 6:
        raise ValueError("matmul accepts at most 6 dimensions")
    return a, b, a_was_1d, b_was_1d


def _restore_rank(out: torch.Tensor, a_was_1d: bool, b_was_1d: bool) -> torch.Tensor:
    if a_was_1d:
        out = out.squeeze(-2)
    if b_was_1d:
        out = out.squeeze(-1)
    return out


def _pack(a: torch.Tensor, b: torch.Tensor, batch, m_dim: int, k_dim: int, n_dim: int, flat: int):
    if flat == 1 and a.ndim == 2 and b.ndim == 2 and a.is_contiguous() and b.is_contiguous():
        return a, b
    a_npu = a.expand(*batch, m_dim, k_dim)
    b_npu = b.expand(*batch, k_dim, n_dim)
    if a_npu.device.type != "npu":
        a_npu = a_npu.to("npu")
    if b_npu.device.type != "npu":
        b_npu = b_npu.to("npu")
    return a_npu.contiguous().reshape(flat * m_dim, k_dim), b_npu.contiguous().reshape(flat * k_dim, n_dim)


@torch.library.custom_op("mojo_npu_catlass_a5::matmul_fwd", mutates_args=())
def matmul_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    source = a.device
    a, b, a_was_1d, b_was_1d = _normalize(a, b)
    if a.dtype != b.dtype or a.dtype not in SUPPORTED_DTYPES:
        raise ValueError("a and b must share float16, bfloat16, or float32")
    if a.shape[-1] != b.shape[-2]:
        raise ValueError("contracted K dimensions differ")

    m_dim = int(a.shape[-2])
    k_dim = int(a.shape[-1])
    n_dim = int(b.shape[-1])
    batch = torch.broadcast_shapes(tuple(a.shape[:-2]), tuple(b.shape[:-2]))
    flat = 1
    for size in batch:
        flat *= int(size)

    out_shape = (*batch, m_dim, n_dim)
    if flat == 0 or m_dim == 0 or n_dim == 0:
        return _restore_rank(a.new_empty(out_shape), a_was_1d, b_was_1d).to(source)
    if k_dim == 0:
        return _restore_rank(a.new_zeros(out_shape), a_was_1d, b_was_1d).to(source)

    a_packed, b_packed = _pack(a, b, batch, m_dim, k_dim, n_dim, flat)
    out = launch(a_packed, b_packed, flat, m_dim, n_dim, k_dim).reshape(out_shape)
    out = _restore_rank(out, a_was_1d, b_was_1d)
    return out if source.type == "npu" else out.to(source)


@matmul_fwd.register_fake
def _matmul_fwd_fake(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a2, b2, a_was_1d, b_was_1d = _normalize(a, b)
    batch = torch.broadcast_shapes(tuple(a2.shape[:-2]), tuple(b2.shape[:-2]))
    return _restore_rank(
        torch.empty(*batch, a2.shape[-2], b2.shape[-1], dtype=a.dtype, device=a.device),
        a_was_1d,
        b_was_1d,
    )
