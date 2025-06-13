"""Tests for the Hessian-Vector Product (HVP) implementation."""

import pytest
import torch
import torch.nn as nn
from torch.nn import functional as F

from torch_secorder.core.hvp import (
    approximate_hvp,
    exact_hvp,
    gauss_newton_product,
    hessian_trace,
    model_hvp,
)


class SimpleModel(nn.Module):
    """A simple model for testing purposes."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 3)
        self.linear2 = nn.Linear(3, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)


def test_exact_hvp_simple_function():
    """Test exact HVP on a simple quadratic function."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    v = torch.tensor([1.0, 1.0])

    def quadratic():
        return x[0] ** 2 + 2 * x[1] ** 2

    hvp = exact_hvp(quadratic, [x], v)

    # For f(x) = x[0]^2 + 2*x[1]^2, the Hessian is [[2, 0], [0, 4]]
    # So Hv should be [2, 4]
    expected = torch.tensor([2.0, 4.0])
    assert torch.allclose(hvp, expected)


def test_exact_hvp_with_model():
    """Test exact HVP with a simple neural network model."""
    model = SimpleModel()
    x = torch.randn(1, 2)
    y = torch.randn(1, 1)
    v = [torch.randn_like(p) for p in model.parameters()]

    def loss_func():
        output = model(x)
        return F.mse_loss(output, y)

    hvp = exact_hvp(loss_func, list(model.parameters()), v)

    # Check that the output has the same structure as the input vector
    assert len(hvp) == len(v)
    for h, vec in zip(hvp, v):
        assert h.shape == vec.shape


def test_approximate_hvp():
    """Test approximate HVP using finite differences."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    v = torch.tensor([1.0, 1.0])

    def quadratic():
        return x[0] ** 2 + 2 * x[1] ** 2

    # Test with different numbers of samples
    for num_samples in [1, 5, 10]:
        hvp = approximate_hvp(quadratic, [x], v, num_samples=num_samples)
        expected = torch.tensor([2.0, 4.0])
        assert torch.allclose(hvp, expected, rtol=5e-2, atol=5e-2)


def test_approximate_hvp_with_damping():
    """Test approximate HVP with damping term."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    v = torch.tensor([1.0, 1.0])
    damping = 0.1

    def quadratic():
        return x[0] ** 2 + 2 * x[1] ** 2

    hvp = approximate_hvp(quadratic, [x], v, damping=damping)
    expected = torch.tensor([2.1, 4.1])  # Hv + damping * v
    assert torch.allclose(hvp, expected, rtol=5e-2, atol=5e-2)


def test_model_hvp():
    """Test the model_hvp convenience function."""
    model = SimpleModel()
    x = torch.randn(1, 2)
    y = torch.randn(1, 1)
    v = [torch.randn_like(p) for p in model.parameters()]

    hvp = model_hvp(model, F.mse_loss, x, y, v)

    # Check that the output has the same structure as the input vector
    assert len(hvp) == len(v)
    for h, vec in zip(hvp, v):
        assert h.shape == vec.shape


def test_hvp_consistency():
    """Test that exact and approximate HVPs give similar results."""
    model = SimpleModel()
    x = torch.randn(1, 2)
    y = torch.randn(1, 1)
    v = [torch.randn_like(p) for p in model.parameters()]

    def loss_func():
        output = model(x)
        return F.mse_loss(output, y)

    exact = exact_hvp(loss_func, list(model.parameters()), v)
    approx = approximate_hvp(loss_func, list(model.parameters()), v, num_samples=200)

    # Check that results are similar
    for e, a in zip(exact, approx):
        assert torch.allclose(e, a, rtol=2e-1, atol=2e-1)


def test_hvp_gradient_flow():
    """Test that HVP computation maintains gradient flow."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    v = torch.tensor([1.0, 1.0])

    def quadratic():
        return x[0] ** 2 + 2 * x[1] ** 2

    hvp = exact_hvp(quadratic, [x], v, create_graph=True)

    # Test that we can compute gradients through the HVP
    loss = hvp.sum()
    grad = torch.autograd.grad(loss, x)[0]
    assert grad is not None
    assert not torch.isnan(grad).any()


def test_hvp_input_validation():
    """Test input validation for HVP functions."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    v = torch.tensor([1.0, 1.0])

    def quadratic():
        return x[0] ** 2 + 2 * x[1] ** 2

    # Test with invalid vector shape
    with pytest.raises(RuntimeError):
        exact_hvp(quadratic, [x], torch.tensor([1.0, 2.0, 3.0]))

    # Test with non-scalar function output
    def non_scalar():
        return torch.tensor([1.0, 2.0])

    with pytest.raises(RuntimeError):
        exact_hvp(non_scalar, [x], v)


def test_gauss_newton_product():
    model = SimpleModel()
    x = torch.randn(1, 2)
    y = model(x).detach()  # Use model output as target to ensure shapes match
    v = [torch.randn_like(p) for p in model.parameters()]
    gn_prod = gauss_newton_product(model, F.mse_loss, x, y, v)
    assert isinstance(gn_prod, list)
    assert len(gn_prod) == len(v)
    for g, p in zip(gn_prod, model.parameters()):
        assert g.shape == p.shape


def test_hessian_trace():
    x = torch.tensor([1.0, 2.0], requires_grad=True)

    def quadratic():
        return x[0] ** 2 + 2 * x[1] ** 2

    trace = hessian_trace(quadratic, [x], num_samples=100)
    expected = 6.0  # Trace of Hessian for quadratic function
    assert abs(trace - expected) < 0.1
