"""Batch-invariant GEMM kernel shared by ``mm`` and ``matmul``.

One Cube Core owns each (batch, M, N) tile and reduces the whole K, so results do
not depend on how the batch is split. Requires the CATLASS Python toolchain, so it is
imported on first use only.
"""

import catlass.tla as tla

from ._tiling import TilingParams


@tla.kernel
def batchinvariant_gemm(
    gm_a: tla.Tensor,
    gm_b: tla.Tensor,
    gm_c: tla.Tensor,
    _tiling: tla.Constexpr[TilingParams],
) -> None:
    """One Cube Core owns each (M, N) tile and reduces the whole K."""
    c0 = 0
    c1 = 1

    dtype_a = gm_a.ptr.dtype
    dtype_b = gm_b.ptr.dtype

    k_dim = gm_a.origin_shape[1]
    n_dim = gm_c.origin_shape[1]
    batch_cnt = gm_b.origin_shape[0] // k_dim
    m_dim = gm_a.origin_shape[0] // batch_cnt

    mutex_l1a0 = tla.mutex(resource="l1a0", id=0)
    mutex_l1a1 = tla.mutex(resource="l1a1", id=1)
    mutex_l1b0 = tla.mutex(resource="l1b0", id=2)
    mutex_l1b1 = tla.mutex(resource="l1b1", id=3)
    mutex_l0a0 = tla.mutex(resource="l0a0", id=4)
    mutex_l0a1 = tla.mutex(resource="l0a1", id=5)
    mutex_l0b0 = tla.mutex(resource="l0b0", id=6)
    mutex_l0b1 = tla.mutex(resource="l0b1", id=7)

    l1a0_ptr = tla.allocate(_tiling.l1_tm * _tiling.l1_tk, dtype_a, tla.AddressSpace.l1, 512)
    l1a1_ptr = tla.allocate(_tiling.l1_tm * _tiling.l1_tk, dtype_a, tla.AddressSpace.l1, 512)
    l1b0_ptr = tla.allocate(_tiling.l1_tk * _tiling.l1_tn, dtype_b, tla.AddressSpace.l1, 512)
    l1b1_ptr = tla.allocate(_tiling.l1_tk * _tiling.l1_tn, dtype_b, tla.AddressSpace.l1, 512)
    l0a0_ptr = tla.allocate(_tiling.l0_tm * _tiling.l0_tk, dtype_a, tla.AddressSpace.l0a, 512)
    l0a1_ptr = tla.allocate(_tiling.l0_tm * _tiling.l0_tk, dtype_a, tla.AddressSpace.l0a, 512)
    l0b0_ptr = tla.allocate(_tiling.l0_tk * _tiling.l0_tn, dtype_b, tla.AddressSpace.l0b, 512)
    l0b1_ptr = tla.allocate(_tiling.l0_tk * _tiling.l0_tn, dtype_b, tla.AddressSpace.l0b, 512)
    l0c_ptr = tla.allocate(_tiling.l0_tm * _tiling.l0_tn, tla.Float32, tla.AddressSpace.l0c, 1024)

    grid_m = (m_dim + _tiling.l1_tm - 1) // _tiling.l1_tm
    grid_n = (n_dim + _tiling.l1_tn - 1) // _tiling.l1_tn
    mn_blocks = grid_m * grid_n
    total_blocks = batch_cnt * mn_blocks

    with tla.cube():
        runtime_zero = tla.as_numeric(0)
        l1_buf_idx = runtime_zero
        l0_buf_idx = runtime_zero
        block_range = tla.range(tla.arch.block_idx(), total_blocks, tla.arch.block_num())
        for loop_idx in block_range:
            batch_idx = loop_idx // mn_blocks
            mn_linear = loop_idx % mn_blocks
            block_row = mn_linear // grid_n
            block_col = mn_linear % grid_n

            # Batch view first, so an M tile cannot cross into the next batch.
            gm_a_batch = tla.tile_view(gm_a, tla.make_shape(m_dim, k_dim), tla.make_coord(batch_idx, c0))
            gm_b_batch = tla.tile_view(gm_b, tla.make_shape(k_dim, n_dim), tla.make_coord(batch_idx, c0))
            gm_c_batch = tla.tile_view(gm_c, tla.make_shape(m_dim, n_dim), tla.make_coord(batch_idx, c0))
            gm_a_by_core = tla.tile_view(
                gm_a_batch,
                tla.make_shape(_tiling.l1_tm, k_dim),
                tla.make_coord(block_row, c0),
            )
            gm_b_by_core = tla.tile_view(
                gm_b_batch,
                tla.make_shape(k_dim, _tiling.l1_tn),
                tla.make_coord(c0, block_col),
            )
            gm_c_by_core = tla.tile_view(
                gm_c_batch,
                tla.make_shape(_tiling.l1_tm, _tiling.l1_tn),
                tla.make_coord(block_row, block_col),
            )

            k_block = gm_a_by_core.origin_shape[1]
            k_l1_count = (k_block + _tiling.l1_tk - 1) // _tiling.l1_tk
            l0_c = tla.make_tensor_like(l0c_ptr, gm_c_by_core)

            for k_l1 in tla.range(c0, k_l1_count, c1):
                gm_a_by_l1 = tla.tile_view(
                    gm_a_by_core,
                    tla.make_shape(_tiling.l1_tm, _tiling.l1_tk),
                    tla.make_coord(c0, k_l1),
                )
                gm_b_by_l1 = tla.tile_view(
                    gm_b_by_core,
                    tla.make_shape(_tiling.l1_tk, _tiling.l1_tn),
                    tla.make_coord(k_l1, c0),
                )
                l1_a = tla.make_tensor_like(l1a0_ptr if (l1_buf_idx == c0) else l1a1_ptr, gm_a_by_l1)
                l1_b = tla.make_tensor_like(l1b0_ptr if (l1_buf_idx == c0) else l1b1_ptr, gm_b_by_l1)
                mutex_l1a = mutex_l1a0 if (l1_buf_idx == c0) else mutex_l1a1
                mutex_l1a.lock(pipe=tla.arch.MTE2)
                tla.copy(l1_a, gm_a_by_l1)
                mutex_l1a.unlock(pipe=tla.arch.MTE2)
                mutex_l1b = mutex_l1b0 if (l1_buf_idx == c0) else mutex_l1b1
                mutex_l1b.lock(pipe=tla.arch.MTE2)
                tla.copy(l1_b, gm_b_by_l1)
                mutex_l1b.unlock(pipe=tla.arch.MTE2)

                k_l0_count = (l1_a.origin_shape[1] + _tiling.l0_tk - 1) // _tiling.l0_tk
                for k_l0 in tla.range(c0, k_l0_count, c1):
                    l1_a_by_l0 = tla.tile_view(
                        l1_a,
                        tla.make_shape(_tiling.l0_tm, _tiling.l0_tk),
                        tla.make_coord(c0, k_l0),
                    )
                    l1_b_by_l0 = tla.tile_view(
                        l1_b,
                        tla.make_shape(_tiling.l0_tk, _tiling.l0_tn),
                        tla.make_coord(k_l0, c0),
                    )
                    l0_a = tla.make_tensor_like(l0a0_ptr if (l0_buf_idx == c0) else l0a1_ptr, l1_a_by_l0)
                    l0_b = tla.make_tensor_like(l0b0_ptr if (l0_buf_idx == c0) else l0b1_ptr, l1_b_by_l0)
                    mutex_l0a = mutex_l0a0 if (l0_buf_idx == c0) else mutex_l0a1
                    mutex_l1a.lock(pipe=tla.arch.MTE1)
                    mutex_l0a.lock(pipe=tla.arch.MTE1)
                    tla.copy(l0_a, l1_a_by_l0)
                    mutex_l0a.unlock(pipe=tla.arch.MTE1)
                    mutex_l1a.unlock(pipe=tla.arch.MTE1)
                    mutex_l0b = mutex_l0b0 if (l0_buf_idx == c0) else mutex_l0b1
                    mutex_l1b.lock(pipe=tla.arch.MTE1)
                    mutex_l0b.lock(pipe=tla.arch.MTE1)
                    tla.copy(l0_b, l1_b_by_l0)
                    mutex_l0b.unlock(pipe=tla.arch.MTE1)
                    mutex_l1b.unlock(pipe=tla.arch.MTE1)

                    mutex_l0a.lock(pipe=tla.arch.CUBE)
                    mutex_l0b.lock(pipe=tla.arch.CUBE)
                    init_c = True if k_l1 == 0 and k_l0 == 0 else False
                    unit_flag = 0b11 if (k_l1 == k_l1_count - 1) and (k_l0 == k_l0_count - 1) else 0b10
                    tla.mmad(l0_c, l0_a, l0_b, init_c=init_c, unit_flag=unit_flag)
                    mutex_l0b.unlock(pipe=tla.arch.CUBE)
                    mutex_l0a.unlock(pipe=tla.arch.CUBE)
                    l0_buf_idx = c1 - l0_buf_idx
                l1_buf_idx = c1 - l1_buf_idx

            tla.copy(
                gm_c_by_core,
                l0_c,
                tla.params.CopyL0C2DstParams(unit_flag=0b11),
            )
