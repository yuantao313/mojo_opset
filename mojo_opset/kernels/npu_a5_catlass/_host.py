"""Host-side plumbing shared by ``mm`` and ``matmul``.

Both operators compile the same kernel and differ only in how they present
batched inputs, so the compile cache, tensor binding and tiling lookup live here.
"""

from functools import lru_cache

import torch

from ._tiling import TilingParams
from ._tiling import tiling_for_shape
from ._toolchain import npu_arch
from ._toolchain import toolchain

SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

_COMPILE_HINT = (
    "CATLASS Python compilation failed. Verify that the compiler artifacts and the BC "
    "templates are ready (`python -m catlass.bc_compile`) and that CANN/torch_npu match the "
    "target, then retry outside any graph capture."
)


def cube_cores(device_index: int) -> int:
    return max(1, int(torch.npu.get_device_properties(int(device_index)).cube_core_num))


@lru_cache(maxsize=64)
def _tiling(m_dim: int, n_dim: int, k_dim: int, dtype: torch.dtype, aic_num: int) -> TilingParams:
    return tiling_for_shape(m_dim, n_dim, k_dim, dtype, aic_num)


@lru_cache(maxsize=64)
def _compiled(m_dim: int, n_dim: int, k_dim: int, batch: int, dtype: torch.dtype, aic_num: int):
    """Compile one kernel per (geometry, batch, dtype, core count)."""

    from . import _kernels

    api = toolchain()
    tiling = _tiling(m_dim, n_dim, k_dim, dtype, aic_num)
    flat_m = batch * m_dim
    flat_k = batch * k_dim
    fakes = (
        api.make_fake_tensor(
            _dsl_dtype(api.tla, dtype),
            (flat_m, k_dim),
            (k_dim, 1),
            layout_tag=api.tla.arch.RowMajor,
            origin_shape=(flat_m, k_dim),
        ),
        api.make_fake_tensor(
            _dsl_dtype(api.tla, dtype),
            (flat_k, n_dim),
            (n_dim, 1),
            layout_tag=api.tla.arch.RowMajor,
            origin_shape=(flat_k, n_dim),
        ),
        api.make_fake_tensor(
            _dsl_dtype(api.tla, dtype),
            (flat_m, n_dim),
            (n_dim, 1),
            layout_tag=api.tla.arch.RowMajor,
            origin_shape=(flat_m, n_dim),
        ),
    )
    try:
        return api.tla.compile(_kernels.batchinvariant_gemm, *fakes, tiling, options=f"--npu-arch {npu_arch()}")
    except Exception as error:
        raise RuntimeError(_COMPILE_HINT) from error


def _dsl_dtype(tla, dtype: torch.dtype):
    return {torch.float16: tla.Float16, torch.bfloat16: tla.BFloat16, torch.float32: tla.Float32}[dtype]


def launch(a: torch.Tensor, b: torch.Tensor, batch: int, m_dim: int, n_dim: int, k_dim: int) -> torch.Tensor:
    """Run the shared kernel over ``batch`` flattened (m, k) x (k, n) pairs."""

    api = toolchain()
    tla = api.tla
    aic_num = cube_cores(a.device.index or 0)
    out = torch.empty(batch * m_dim, n_dim, dtype=a.dtype, device="npu")
    compiled = _compiled(m_dim, n_dim, k_dim, batch, a.dtype, aic_num)
    tla_a = api.from_dlpack(a, layout_tag=tla.arch.RowMajor)
    tla_b = api.from_dlpack(b, layout_tag=tla.arch.RowMajor)
    tla_c = api.from_dlpack(out, layout_tag=tla.arch.RowMajor)
    compiled(tla_a, tla_b, tla_c, block_num=aic_num)
    return out
