"""Lazy access to the CATLASS Python toolchain."""

import os

from functools import lru_cache
from typing import NamedTuple
from typing import Optional

from mojo_opset.utils.target import Target
from mojo_opset.utils.target import TargetLike
from mojo_opset.utils.target import detect_target

_SUPPORTED_ARCHES = frozenset({"a5"})

_DEFAULT_NPU_ARCH = "3510"

_CACHE_ENV = "CATLASS_DSL_CACHE_DIR"
_CACHE_ENV_ALIAS = "MOJO_CATLASS_CACHE_DIR"
_ARCH_ENV = "MOJO_CATLASS_NPU_ARCH"

_INSTALL_HINT = (
    "CATLASS Python is unavailable. It requires an Ascend A5 generation device (950PR/950DT), CANN>=9.2.0 and "
    "Python <3.14, plus the `ascend-catlass-dsl` package. Install it with "
    "`pip install mojo-opset[catlass]` and pre-compile the BC templates once with "
    "`python -m catlass.bc_compile`."
)


class Toolchain(NamedTuple):
    tla: object
    from_dlpack: object
    make_fake_tensor: object


def require_supported_target(target: Optional[TargetLike] = None) -> Target:
    resolved = detect_target(target)
    if resolved.platform != "npu" or resolved.arch not in _SUPPORTED_ARCHES:
        raise RuntimeError(
            f"CATLASS Python supports the Ascend A5 generation (950PR/950DT) only, "
            f"got target={resolved.key!r}. Select a supported target or another implementation."
        )
    return resolved


def npu_arch() -> str:
    return os.getenv(_ARCH_ENV, _DEFAULT_NPU_ARCH)


@lru_cache(maxsize=1)
def toolchain() -> Toolchain:
    _configure_cache_dir()
    import torch_npu  # noqa: F401  -- the runtime links against the vendor libraries

    try:
        from catlass import tla
        from catlass.tla.runtime import from_dlpack
        from catlass.tla.runtime import make_fake_tensor
    except ImportError as error:
        raise ImportError(_INSTALL_HINT) from error
    return Toolchain(tla=tla, from_dlpack=from_dlpack, make_fake_tensor=make_fake_tensor)


def _configure_cache_dir() -> None:
    if os.getenv(_CACHE_ENV):
        return
    root = os.getenv(_CACHE_ENV_ALIAS)
    if root:
        os.environ[_CACHE_ENV] = root
