"""``mm`` compiled from a batch-invariant CATLASS Python kernel."""

import torch

from ._host import SUPPORTED_DTYPES
from ._host import launch


@torch.library.custom_op("mojo_npu_catlass_a5::mm_fwd", mutates_args=())
def mm_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("mm expects two rank-2 tensors")
    if a.dtype != b.dtype or a.dtype not in SUPPORTED_DTYPES:
        raise ValueError("a and b must share float16, bfloat16, or float32")
    if a.shape[-1] != b.shape[-2]:
        raise ValueError("contracted K dimensions differ")

    m_dim = int(a.shape[0])
    k_dim = int(a.shape[1])
    n_dim = int(b.shape[1])
    source = a.device
    if m_dim == 0 or n_dim == 0:
        return a.new_empty(m_dim, n_dim).to(source)
    if k_dim == 0:
        return a.new_zeros(m_dim, n_dim).to(source)

    a_packed = a.contiguous() if a.device.type == "npu" else a.to("npu").contiguous()
    b_packed = b.contiguous() if b.device.type == "npu" else b.to("npu").contiguous()
    out = launch(a_packed, b_packed, 1, m_dim, n_dim, k_dim).reshape(m_dim, n_dim)
    return out if source.type == "npu" else out.to(source)


@mm_fwd.register_fake
def _mm_fwd_fake(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.empty(a.shape[0], b.shape[1], dtype=a.dtype, device=a.device)
