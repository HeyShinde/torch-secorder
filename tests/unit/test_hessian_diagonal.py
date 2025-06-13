"""Tests for the Hessian diagonal computation functions."""

import pytest
import torch
from torch import nn

from torch_secorder.core.hessian_diagonal import (
    hessian_diagonal,
    model_hessian_diagonal,
)


def test_hessian_diagonal_quadratic():
    """Test Hessian diagonal computation on a simple quadratic function."""
    # Create a simple quadratic function: f(x) = x^T A x
    A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
    x = torch.tensor([1.0, 2.0], requires_grad=True)

    def quadratic():
        return x @ A @ x

    diag = hessian_diagonal(quadratic, [x])
    expected_diag = [torch.tensor([4.0, 6.0])]  # Diagonal of 2A

    assert len(diag) == 1
    assert torch.allclose(diag[0], expected_diag[0])


def test_model_hessian_diagonal():
    """Test Hessian diagonal computation on a simple neural network."""
    model = nn.Linear(2, 1)
    x = torch.randn(5, 2)
    y = torch.randn(5, 1)

    diag = model_hessian_diagonal(model, nn.functional.mse_loss, x, y)

    # Check that we get diagonal elements for each parameter
    assert len(diag) == 2  # One for weight, one for bias
    assert diag[0].shape == model.weight.shape
    assert diag[1].shape == model.bias.shape


def test_hessian_diagonal_with_custom_vectors():
    """Test Hessian diagonal computation with custom vectors."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    v = [torch.tensor([2.0, 3.0])]

    def quadratic():
        return x.pow(2).sum()

    diag = hessian_diagonal(quadratic, [x], v=v)
    expected_diag = [torch.tensor([4.0, 6.0])]  # 2 * v

    assert len(diag) == 1
    assert torch.allclose(diag[0], expected_diag[0])


def test_hessian_diagonal_create_graph():
    """Test Hessian diagonal computation with create_graph=True."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)

    def cubic():
        return x.pow(3).sum()

    diag = hessian_diagonal(cubic, [x], create_graph=True)

    # Check that we can compute gradients of the diagonal
    grad = torch.autograd.grad(diag[0].sum(), x)[0]
    assert grad is not None


def test_hessian_diagonal_strict():
    """Test Hessian diagonal computation with strict mode in all scenarios."""
    # 1. strict=False, no parameter requires grad (should pass, return zeros)
    print(
        "\nTest: strict=False, no parameter requires grad (should pass, return zeros)"
    )
    a = torch.tensor([1.0, 2.0])  # requires_grad=False by default

    def func_a():
        return a.sum()

    diag = hessian_diagonal(func_a, [a], strict=False)
    assert torch.allclose(diag[0], torch.zeros_like(a))

    # 2. strict=True, all parameters require grad (should pass)
    print("\nTest: strict=True, all parameters require grad (should pass)")
    b = torch.tensor([1.0, 2.0], requires_grad=True)

    def func_b():
        return (b * b).sum()

    diag = hessian_diagonal(func_b, [b], strict=True)
    assert torch.allclose(diag[0], torch.full_like(b, 2.0))

    # 3. strict=True, at least one parameter does NOT require grad (should raise RuntimeError)
    print(
        "\nTest: strict=True, at least one parameter does NOT require grad (should raise RuntimeError)"
    )
    c = torch.tensor([1.0, 2.0], requires_grad=True)
    d = torch.tensor([3.0, 4.0])  # requires_grad=False

    def func_cd():
        return (c * d).sum()

    with pytest.raises(
        RuntimeError, match="One of the differentiated Tensors does not require grad"
    ):
        hessian_diagonal(func_cd, [c, d], strict=True)

    # 4. strict=False, mixed parameters (should pass, zeros for non-grad)
    print("\nTest: strict=False, mixed parameters (should pass, zeros for non-grad)")
    diag = hessian_diagonal(func_cd, [c, d], strict=False)
    assert torch.allclose(diag[0], torch.zeros_like(c)) or torch.allclose(
        diag[0], torch.full_like(c, 0.0)
    )
    assert torch.allclose(diag[1], torch.zeros_like(d))
