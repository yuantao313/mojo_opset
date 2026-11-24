"""Tests for functions.mm and functions.matmul."""

from functools import partial

import pytest
import torch

from mojo_opset import functions as F
from mojo_opset import preload
from tests._checks import assert_close
from tests._checks import assert_repeatable

MM_SHAPES = [(16, 16, 16), (17, 31, 48), (64, 64, 64), (65, 63, 67), (128, 96, 80)]

MATMUL_SHAPES = [
    ((64, 64), (64, 64)),
    ((17, 31), (31, 48)),
    ((4, 64, 32), (32, 16)),
    ((512,), (512, 256)),
    ((256, 512), (512,)),
    ((2, 3, 64, 32), (32, 16)),
]


@pytest.mark.reference
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_mm_reference_matches_cpu(dtype):
    a = torch.rand(32, 48, dtype=dtype) - 0.5
    b = torch.rand(48, 16, dtype=dtype) - 0.5
    actual = F.mm(a, b, implementation="torch")
    assert_close(actual, torch.mm(a, b), dtype)


@pytest.mark.reference
@pytest.mark.accuracy
def test_mm_reference_rejects_rank():
    with pytest.raises(ValueError, match="rank-2"):
        F.mm(torch.zeros(4, 8, 2), torch.zeros(8, 8), implementation="torch")


@pytest.mark.reference
@pytest.mark.accuracy
def test_mm_reference_rejects_requires_no_grad():
    a = torch.randn(16, 16, requires_grad=True)
    b = torch.randn(16, 16)
    with pytest.raises(RuntimeError, match="does not support autograd"):
        F.mm(a, b, implementation="torch")


@pytest.mark.reference
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_matmul_reference_matches_cpu(dtype):
    a = torch.rand(4, 8, 16, dtype=dtype) - 0.5
    b = torch.rand(16, 12, dtype=dtype) - 0.5
    actual = F.matmul(a, b, implementation="torch")
    assert_close(actual, torch.matmul(a, b), dtype)


@pytest.mark.api("functions.mm")
@pytest.mark.parametrize("shape", MM_SHAPES)
@pytest.mark.accuracy
def test_mm_accuracy(accuracy_backend, shape):
    implementation, _, device = accuracy_backend
    m, k, n = shape
    a = torch.rand(m, k, device=device, dtype=torch.float32) - 0.5
    b = torch.rand(k, n, device=device, dtype=torch.float32) - 0.5
    actual = F.mm(a, b, implementation=implementation)
    expected = F.mm(a, b, implementation="torch")
    assert_close(actual, expected, torch.float32)
    assert actual.shape == (m, n)


@pytest.mark.api("functions.mm")
@pytest.mark.parametrize("shape", MM_SHAPES[:3])
@pytest.mark.bitwise
def test_mm_bitwise(accuracy_backend, shape):
    implementation, _, device = accuracy_backend
    m, k, n = shape
    a = torch.rand(m, k, device=device, dtype=torch.float32) - 0.5
    b = torch.rand(k, n, device=device, dtype=torch.float32) - 0.5
    assert_repeatable(partial(F.mm, implementation=implementation), (a, b), backward=False)


@pytest.mark.api("functions.matmul")
@pytest.mark.parametrize("shapes", MATMUL_SHAPES)
@pytest.mark.accuracy
def test_matmul_accuracy(accuracy_backend, shapes):
    implementation, _, device = accuracy_backend
    a_shape, b_shape = shapes
    a = torch.rand(*a_shape, device=device, dtype=torch.float32) - 0.5
    b = torch.rand(*b_shape, device=device, dtype=torch.float32) - 0.5
    actual = F.matmul(a, b, implementation=implementation)
    expected = F.matmul(a, b, implementation="torch")
    assert_close(actual, expected, torch.float32)
    assert actual.shape == expected.shape


@pytest.mark.api("functions.mm")
@pytest.mark.accuracy
def test_mm_fake_metadata(accuracy_backend):
    implementation, target, device = accuracy_backend
    from mojo_opset import Target
    from mojo_opset.functions._dispatch import resolve_implementation

    selected = implementation or resolve_implementation("mm", Target.parse(target))
    if selected != "catlass":
        pytest.skip("only the CATLASS provider exposes a fake-registered custom op")
    preload("mm", implementation="catlass", target=target)
    from mojo_opset.kernels.npu_a5_catlass.mm import mm_fwd

    a = torch.empty(32, 48, device=device, dtype=torch.float32)
    b = torch.empty(48, 16, device=device, dtype=torch.float32)
    torch.library.opcheck(mm_fwd, (a, b), test_utils=("test_faketensor",))
