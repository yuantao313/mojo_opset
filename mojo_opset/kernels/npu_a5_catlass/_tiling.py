"""Tiling for the batch-invariant GEMM kernels.

Port of the CATLASS Python reference tiling (AswTiling on DAV_3510): the L1/L0
tile shape depends only on one GEMM's M, N, K, dtype and Cube core count, never
on the batch count. ``mm`` and ``matmul`` share it.
"""

import math

from dataclasses import dataclass

import catlass.tla as tla
import torch


@dataclass(frozen=True)
class TilingParams:
    l1_tm: tla.Constexpr[int] = 256
    l1_tn: tla.Constexpr[int] = 256
    l1_tk: tla.Constexpr[int] = 128
    l0_tm: tla.Constexpr[int] = 256
    l0_tn: tla.Constexpr[int] = 256
    l0_tk: tla.Constexpr[int] = 32


_L0A_SIZE = 64 * 1024
_L1_SIZE = 512 * 1024
_L0C_SIZE = 256 * 1024
_DB = 2
_ALIGN16 = 16
_ALIGN64 = 64
_ALIGN128 = 128
_ALIGN256 = 256
_MIN_BASE = 112
_MAX_BASE = 336
_LOAD_BALANCE_LIMIT = 0.88
_MAX_SINGLE_CORE_ROUND = 10.0
_LOAD_BALANCING_THRESHOLD = 0.98
_BALANCE_REDUNDANT_THRESHOLD = 0.97
_MIN_EQUALIZATION = 1.02
_MAX_LOOP = 20
_K128_BYTE = 128
_K256_BYTE = 256

# Load-balance table entries: (block on the larger MN side, the other MN side, K, cost).
_BLOCK_TABLE = (
    (336, 192, 48, 1.0),
    (336, 176, 48, 1.0),
    (336, 160, 48, 1.0),
    (336, 144, 48, 1.073),
    (336, 128, 48, 1.167),
    (336, 112, 48, 1.287),
    (320, 192, 48, 1.0),
    (320, 176, 48, 1.0),
    (320, 160, 48, 1.014),
    (320, 144, 48, 1.089),
    (320, 128, 48, 1.183),
    (304, 208, 48, 1.0),
    (304, 192, 48, 1.0),
    (304, 176, 48, 1.0),
    (304, 160, 48, 1.032),
    (304, 144, 48, 1.107),
    (304, 128, 48, 1.201),
    (288, 224, 48, 1.0),
    (288, 208, 48, 1.0),
    (288, 192, 48, 1.0),
    (288, 176, 48, 1.0),
    (288, 160, 48, 1.051),
    (288, 144, 48, 1.126),
    (288, 128, 48, 1.220),
    (272, 240, 48, 1.0),
    (272, 224, 48, 1.0),
    (272, 208, 48, 1.0),
    (272, 192, 48, 1.0),
    (272, 176, 48, 1.012),
    (272, 160, 48, 1.073),
    (272, 144, 48, 1.148),
    (272, 128, 48, 1.242),
    (256, 256, 64, 1.0),
    (256, 240, 64, 1.0),
    (256, 224, 64, 1.0),
    (256, 208, 64, 1.0),
    (256, 192, 64, 1.0),
    (256, 176, 64, 1.037),
    (256, 160, 64, 1.098),
    (256, 144, 64, 1.173),
    (256, 128, 64, 1.267),
    (240, 272, 48, 1.0),
    (240, 256, 64, 1.0),
    (240, 240, 64, 1.0),
    (240, 224, 64, 1.0),
    (240, 208, 64, 1.0),
    (240, 192, 64, 1.014),
    (240, 176, 64, 1.065),
    (240, 160, 64, 1.126),
    (240, 144, 64, 1.201),
    (240, 128, 64, 1.295),
    (224, 288, 48, 1.0),
    (224, 272, 48, 1.0),
    (224, 256, 64, 1.0),
    (224, 240, 64, 1.0),
    (224, 224, 64, 1.0),
    (224, 208, 64, 1.003),
    (224, 192, 64, 1.046),
    (224, 176, 64, 1.097),
    (224, 160, 64, 1.159),
    (224, 144, 64, 1.234),
    (208, 304, 48, 1.0),
    (208, 288, 48, 1.0),
    (208, 272, 48, 1.0),
    (208, 256, 64, 1.0),
    (208, 240, 64, 1.0),
    (208, 224, 64, 1.003),
    (208, 208, 64, 1.04),
    (208, 192, 64, 1.083),
    (208, 176, 64, 1.134),
    (208, 160, 64, 1.196),
    (208, 144, 64, 1.271),
    (192, 336, 48, 1.0),
    (192, 320, 48, 1.0),
    (192, 304, 48, 1.0),
    (192, 288, 48, 1.0),
    (192, 272, 48, 1.0),
    (192, 256, 64, 1.0),
    (192, 240, 64, 1.014),
    (192, 224, 64, 1.046),
    (192, 208, 64, 1.083),
    (192, 192, 80, 1.126),
    (192, 176, 80, 1.178),
    (192, 160, 80, 1.239),
    (176, 336, 48, 1.0),
    (176, 320, 48, 1.0),
    (176, 304, 48, 1.0),
    (176, 288, 48, 1.0),
    (176, 272, 48, 1.012),
    (176, 256, 64, 1.037),
    (176, 240, 64, 1.065),
    (176, 224, 64, 1.097),
    (176, 208, 64, 1.134),
    (176, 192, 80, 1.178),
    (176, 176, 80, 1.229),
    (176, 160, 80, 1.290),
    (160, 336, 48, 1.0),
    (160, 320, 48, 1.014),
    (160, 304, 48, 1.032),
    (160, 288, 48, 1.051),
    (160, 272, 48, 1.073),
    (160, 256, 64, 1.098),
    (160, 240, 64, 1.126),
    (160, 224, 64, 1.159),
    (160, 208, 64, 1.196),
    (160, 192, 80, 1.239),
    (160, 176, 80, 1.290),
    (144, 336, 48, 1.073),
    (144, 320, 48, 1.089),
    (144, 304, 48, 1.107),
    (144, 288, 48, 1.126),
    (144, 272, 48, 1.148),
    (144, 256, 64, 1.173),
    (144, 240, 64, 1.201),
    (144, 224, 64, 1.234),
    (144, 208, 64, 1.271),
    (128, 336, 48, 1.167),
    (128, 320, 48, 1.183),
    (128, 304, 48, 1.201),
    (128, 288, 48, 1.220),
    (128, 272, 48, 1.242),
    (128, 256, 64, 1.267),
    (128, 240, 64, 1.295),
    (112, 336, 48, 1.287),
)
_BLOCK_LOOKUP = {
    336: (6, 0),
    320: (5, 6),
    304: (6, 11),
    288: (7, 17),
    272: (8, 24),
    256: (9, 32),
    240: (10, 41),
    224: (10, 51),
    208: (11, 61),
    192: (12, 72),
    176: (12, 84),
    160: (11, 96),
    144: (9, 107),
    128: (7, 116),
    112: (1, 123),
}


def _ceil_div(value: int, divisor: int) -> int:
    if divisor <= 0:
        return 0
    return (int(value) + int(divisor) - 1) // int(divisor)


def _ceil_align(value: int, align: int) -> int:
    return _ceil_div(value, align) * align


def _floor_align(value: int, align: int) -> int:
    if align <= 0:
        return int(value)
    return (int(value) // int(align)) * int(align)


def _elem_bytes(dtype: torch.dtype) -> int:
    return 4 if dtype == torch.float32 else 2


def _multi_core_balance(m: int, n: int, aic_num: int, base_m: int, base_n: int) -> float:
    if base_m <= 0 or base_n <= 0 or aic_num <= 0:
        return 0.0
    m_cnt = _ceil_div(m, base_m)
    n_cnt = _ceil_div(n, base_n)
    m_tail = m % base_m
    n_tail = n % base_n
    m_main = m // base_m
    n_main = n // base_n
    total = m_cnt * n_cnt
    total_main = m_main * n_main
    if total == 0:
        return 0.0
    avg = (m * n) / aic_num
    single_max_tail = max(base_m * n_tail, base_n * m_tail)
    core_tail = total % aic_num
    tail_ratio = aic_num // core_tail if core_tail else 1

    def _div(base: int) -> int:
        if base % _ALIGN256 == 0:
            return _ALIGN16
        if base % _ALIGN128 == 0:
            return _ALIGN16 // 2
        if base % _ALIGN64 == 0:
            return 4
        if base % 32 == 0:
            return 2
        return 1

    tail_ratio = min(tail_ratio, _div(base_m) * _div(base_n))
    tail_num = _ceil_div(total, aic_num) - _ceil_div(total_main, aic_num)
    no_div_tail = tail_num - 1 if tail_num >= 2 else 0
    if tail_num == 0:
        extra = _ceil_div(total_main, aic_num) - total_main // aic_num
        max_load = (total_main // aic_num) * base_m * base_n + extra * base_m * base_n / tail_ratio
    else:
        max_load = (
            _ceil_div(total_main, aic_num) * base_m * base_n
            + no_div_tail * single_max_tail
            + (tail_num - no_div_tail) * single_max_tail / tail_ratio
        )
    if max_load <= 0:
        return 0.0
    return avg / max_load


def _redundant_move(base_m: int, base_n: int, m: int, n: int) -> int:
    return _ceil_div(m, base_m) * n + _ceil_div(n, base_n) * m - (m + n)


def _calc_basic_block(m: int, n: int, base_m: int, base_n: int, used_cores: int) -> tuple[int, int]:
    m_core = _ceil_div(m, base_m)
    n_core = _ceil_div(n, base_n)
    if m_core == 0 or n_core == 0 or used_cores <= 0:
        return base_m, base_n
    if m_core <= n_core:
        base_m = _ceil_align(_ceil_div(m, m_core), _ALIGN16)
        m_core = _ceil_div(m, base_m)
        n_core = max(used_cores // max(m_core, 1), 1)
        base_n = _ceil_align(_ceil_div(n, n_core), _ALIGN16)
    else:
        base_n = _ceil_align(_ceil_div(n, n_core), _ALIGN16)
        n_core = _ceil_div(n, base_n)
        m_core = max(used_cores // max(n_core, 1), 1)
        base_m = _ceil_align(_ceil_div(m, m_core), _ALIGN16)
    while base_n >= base_m * 2 and n_core < used_cores // 2:
        n_core = n_core * 2
        m_core = max(used_cores // n_core, 1)
        base_m = _ceil_align(_ceil_div(m, m_core), _ALIGN16)
        base_n = _ceil_align(_ceil_div(n, n_core), _ALIGN16)
        m_core = _ceil_div(m, base_m)
        n_core = _ceil_div(n, base_n)
        if m_core <= 0 or n_core <= 0:
            break
    while base_m >= base_n * 2 and m_core < used_cores // 2:
        m_core = m_core * 2
        n_core = max(used_cores // m_core, 1)
        base_m = _ceil_align(_ceil_div(m, m_core), _ALIGN16)
        base_n = _ceil_align(_ceil_div(n, n_core), _ALIGN16)
        m_core = _ceil_div(m, base_m)
        n_core = _ceil_div(n, base_n)
        if m_core <= 0 or n_core <= 0:
            break
    return max(base_m, _ALIGN16), max(base_n, _ALIGN16)


def _update_base_block(base_block: int, is_m_larger: bool, dtype_size: int, k: int) -> int:
    if not is_m_larger:
        align = _ALIGN128 // dtype_size if k <= _ALIGN256 else _ALIGN64 // dtype_size
        return _ceil_align(base_block, max(align, 1))
    return _ceil_align(base_block, _ALIGN16)


def _handle_large_single_side(
    m: int,
    n: int,
    min_mn: int,
    max_mn: int,
    is_m_larger: bool,
    aic_num: int,
    dtype_size: int,
    k: int,
) -> tuple[int, int]:
    db_l0c = 1
    if is_m_larger:
        base_n = min_mn
        base_m = _floor_align(_L0C_SIZE // db_l0c // max(base_n, 1) // 4, _ALIGN16)
        target = base_m
    else:
        base_m = min_mn
        base_n = _floor_align(_L0C_SIZE // db_l0c // max(base_m, 1) // 4, 32)
        target = base_n
    min_cores = (aic_num + 1) * 9 // 10
    found = target
    for tmp_cores in range(aic_num, min_cores - 1, -1):
        for loop in range(1, _MAX_LOOP + 1):
            block = _update_base_block(_ceil_div(max_mn, tmp_cores * loop), is_m_larger, dtype_size, k)
            tile = block * min_mn * 4
            if _MIN_BASE <= block <= _MAX_BASE and tile <= _L0C_SIZE:
                found = block
                if is_m_larger:
                    return max(found, _ALIGN16), max(base_n, _ALIGN16)
                return max(base_m, _ALIGN16), max(found, _ALIGN16)
    if is_m_larger:
        return max(base_m, _ALIGN16), max(base_n, _ALIGN16)
    return max(base_m, _ALIGN16), max(base_n, _ALIGN16)


def _n_not_aligned(base_n: int, dtype_size: int, k: int) -> bool:
    if k <= _ALIGN256:
        return base_n * dtype_size % _ALIGN128 != 0
    return base_n * dtype_size % _ALIGN64 != 0


def _handle_large_both_sides(
    m: int,
    n: int,
    k: int,
    aic_num: int,
    dtype_size: int,
    is_fp32: bool,
    is_m_larger: bool,
    higher_x: int,
    lower_x: int,
    min_mn: int,
    default_balance: float,
    redundant: int,
) -> tuple[int, int]:
    init_k = _K128_BYTE // dtype_size
    best = {
        "balance": default_balance,
        "m": _K256_BYTE,
        "n": _K256_BYTE,
        "k": init_k,
        "hit": False,
    }

    def search(start: int, end: int, negative: bool) -> bool:
        base_x = start
        while (base_x >= end) if negative else (base_x <= end):
            lookup = _BLOCK_LOOKUP.get(base_x)
            if lookup is None:
                base_x = base_x - _ALIGN16 if negative else base_x + _ALIGN16
                continue
            count, start_index = lookup
            for i in range(count):
                x1, x2, x3, x4 = _BLOCK_TABLE[start_index + i]
                if x3 > k:
                    continue
                if is_fp32 and (x3 // 2) % _ALIGN16 != 0:
                    continue
                cur_m = x2 if is_m_larger else x1
                cur_n = x1 if is_m_larger else x2
                if _n_not_aligned(cur_n, dtype_size, k):
                    continue
                if cur_m * dtype_size % _ALIGN64 != 0:
                    continue
                balance = _multi_core_balance(m, n, aic_num, cur_m, cur_n) / x4
                remove = _redundant_move(cur_m, cur_n, m, n) / max(redundant, 1)
                if balance - default_balance <= remove - _BALANCE_REDUNDANT_THRESHOLD:
                    continue
                if balance > _LOAD_BALANCING_THRESHOLD:
                    best["m"] = cur_m
                    best["n"] = cur_n
                    best["hit"] = True
                    return True
                if balance > best["balance"] and balance > _MIN_EQUALIZATION * default_balance:
                    best["balance"] = balance
                    best["m"] = cur_m
                    best["n"] = cur_n
                    best["hit"] = True
            base_x = base_x - _ALIGN16 if negative else base_x + _ALIGN16
        return False

    if search(lower_x, _MIN_BASE, True):
        return best["m"], best["n"]
    second = dict(best)
    if search(higher_x, min(min_mn, _MAX_BASE), False):
        return best["m"], best["n"]
    if second["hit"] or best["hit"]:
        pick = second if second["balance"] >= best["balance"] else best
        return pick["m"], pick["n"]
    return best["m"], best["n"]


def _cal_l1_stepk(m: int, n: int, k: int, base_m: int, base_n: int, base_k: int, dtype_size: int) -> int:
    half_l1 = _L1_SIZE // 2
    depth_a = max(half_l1 // max(base_m * base_k * dtype_size, 1), 1)
    depth_b = max(half_l1 // max(base_n * base_k * dtype_size, 1), 1)
    if depth_a * base_m * base_k * dtype_size + depth_b * base_n * base_k * dtype_size > half_l1 * 2:
        if base_m <= base_n:
            depth_a = max(depth_a // 2, 1)
        else:
            depth_b = max(depth_b // 2, 1)
    step_ka = max(depth_a // _DB, 1)
    step_kb = max(depth_b // _DB, 1)
    if (
        base_m == _ALIGN256
        and base_n == _ALIGN256
        and m % _ALIGN16 == 0
        and n % _ALIGN16 == 0
        and k % _ALIGN16 == 0
        and k <= _ALIGN256
    ):
        step_ka = min(step_ka, 2)
        step_kb = min(step_kb, 2)
    if step_ka >= step_kb:
        step_ka = max(step_ka // step_kb * step_kb, 1)
    else:
        step_kb = max(step_kb // step_ka * step_ka, 1)
    return min(step_ka, step_kb)


def tiling_for_shape(m: int, n: int, k: int, dtype: torch.dtype, aic_num: int) -> TilingParams:
    """Select L1/L0 tiles for one GEMM on DAV_3510. Batch count is not an input."""
    m = max(int(m), 1)
    n = max(int(n), 1)
    k = max(int(k), 1)
    aic_num = max(int(aic_num), 1)
    dtype_size = _elem_bytes(dtype)
    is_fp32 = dtype == torch.float32
    base_m = _ALIGN256
    base_n = _ALIGN256
    used_cores = aic_num
    base_m = min(_ceil_align(m, _ALIGN16), base_m)
    base_n = min(_ceil_align(n, _ALIGN16), base_n)
    default_balance = _multi_core_balance(m, n, aic_num, base_m, base_n)
    redundant = max(_redundant_move(base_m, base_n, m, n), 1)
    m_core = _ceil_div(m, base_m)
    n_core = _ceil_div(n, base_n)
    single_blocks = (m_core * n_core) // aic_num
    if 1.0 <= single_blocks <= _MAX_SINGLE_CORE_ROUND and default_balance < _LOAD_BALANCE_LIMIT:
        data = float(m * n) / aic_num
        while data > float(_ALIGN256 * _ALIGN256):
            data /= 2.0
        best = math.sqrt(data)
        higher = _ceil_align(int(math.ceil(best)), _ALIGN16)
        lower = _floor_align(int(math.floor(best)), _ALIGN16)
        min_mn = _ceil_align(min(m, n), _ALIGN16)
        max_mn = _ceil_align(max(m, n), _ALIGN16)
        is_m_larger = m > n
        if lower >= min_mn:
            base_m, base_n = _handle_large_single_side(m, n, min_mn, max_mn, is_m_larger, aic_num, dtype_size, k)
        else:
            base_m, base_n = _handle_large_both_sides(
                m,
                n,
                k,
                aic_num,
                dtype_size,
                is_fp32,
                is_m_larger,
                higher,
                lower,
                min_mn,
                default_balance,
                redundant,
            )
    if single_blocks < 1:
        base_m, base_n = _calc_basic_block(m, n, base_m, base_n, used_cores)
    base_m = max(_ceil_align(base_m, _ALIGN16), _ALIGN16)
    base_n = max(_ceil_align(base_n, _ALIGN16), _ALIGN16)
    k_align = _ALIGN16
    k_max = _floor_align(_L0A_SIZE // _DB // dtype_size // max(base_m, base_n), k_align)
    base_k = min(_ceil_align(k, k_align), max(k_max, k_align))
    step_k = _cal_l1_stepk(m, n, k, base_m, base_n, base_k, dtype_size)
    l1_tk = max(step_k * base_k, base_k)
    l1_tk = min(l1_tk, max(_ceil_align(k, base_k), base_k))
    l0a = base_m * base_k * dtype_size * _DB
    l0b = base_k * base_n * dtype_size * _DB
    l0c = base_m * base_n * 4
    l1 = (base_m * l1_tk + l1_tk * base_n) * dtype_size * _DB
    while (l0a > _L0A_SIZE or l0b > _L0A_SIZE or l0c > _L0C_SIZE or l1 > _L1_SIZE) and base_k > _ALIGN16:
        base_k = max(_floor_align(base_k // 2, _ALIGN16), _ALIGN16)
        step_k = _cal_l1_stepk(m, n, k, base_m, base_n, base_k, dtype_size)
        l1_tk = max(step_k * base_k, base_k)
        l0a = base_m * base_k * dtype_size * _DB
        l0b = base_k * base_n * dtype_size * _DB
        l1 = (base_m * l1_tk + l1_tk * base_n) * dtype_size * _DB
    while l1 > _L1_SIZE and l1_tk > base_k:
        l1_tk -= base_k
        l1 = (base_m * l1_tk + l1_tk * base_n) * dtype_size * _DB
    l1_tk = min(max(l1_tk, base_k), max(_ceil_align(k, base_k), base_k))
    return TilingParams(
        l1_tm=base_m,
        l1_tn=base_n,
        l1_tk=l1_tk,
        l0_tm=base_m,
        l0_tn=base_n,
        l0_tk=base_k,
    )
