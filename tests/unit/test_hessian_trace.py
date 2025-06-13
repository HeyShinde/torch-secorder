"""Tests for the Hessian trace computation functions."""

import torch
from torch import nn

from torch_secorder.core.hessian_trace import hessian_trace, model_hessian_trace


def test_hessian_trace_quadratic():
    """Test Hessian trace computation on a simple quadratic function."""
    # Create a simple quadratic function: f(x) = x^T A x
    A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
    x = torch.tensor([1.0, 2.0], requires_grad=True)

    def quadratic():
        return x @ A @ x

    trace = hessian_trace(quadratic, [x])
    expected_trace = 10.0  # Trace of 2A

    assert torch.allclose(trace, torch.tensor(expected_trace), rtol=0.1)


def test_model_hessian_trace():
    """Test Hessian trace computation on a simple neural network."""
    model = nn.Linear(2, 1)
    x = torch.randn(5, 2)
    y = torch.randn(5, 1)

    trace = model_hessian_trace(model, nn.functional.mse_loss, x, y)

    # Check that we get a scalar trace
    assert isinstance(trace, torch.Tensor)
    assert trace.ndim == 0


def test_hessian_trace_convergence():
    """Test that Hessian trace estimation converges with more samples."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)

    def quadratic():
        return x.pow(2).sum()

    # Compute trace with different numbers of samples
    trace1 = hessian_trace(quadratic, [x], num_samples=10)
    trace2 = hessian_trace(quadratic, [x], num_samples=1000)

    # The estimate with more samples should be closer to the true value (4.0)
    assert abs(trace2 - 4.0) <= abs(trace1 - 4.0)
