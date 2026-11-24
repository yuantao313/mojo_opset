"""CATLASS Python provider for the Ascend A5 generation."""

from mojo_opset.utils._exports import extend_bindings as _extend_bindings

OPS = {"mm": "mm", "matmul": "matmul"}


def prepare(op_id, target, implementation):
    del op_id, implementation
    from ._toolchain import require_supported_target
    from ._toolchain import toolchain

    require_supported_target(target)
    toolchain()


_extend_bindings(__name__, OPS)
del _extend_bindings
