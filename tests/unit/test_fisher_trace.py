"""Tests for the Fisher trace computation functions."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_secorder.approximations.fisher_trace import (
    empirical_fisher_trace,
    generalized_fisher_trace,
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 3)  # 3 classes

    def forward(self, x):
        return self.linear(x)


def test_empirical_fisher_trace_mse():
    """Test EFIM trace computation with MSE loss."""
    model = SimpleModel()
    inputs = torch.randn(5, 10)
    targets = torch.randn(5, 1)

    efim_trace = empirical_fisher_trace(model, nn.MSELoss(), inputs, targets)

    assert isinstance(efim_trace, torch.Tensor)
    assert efim_trace.ndim == 0  # Should be a scalar

    # Manual computation: sum of squared gradients
    loss = nn.MSELoss()(model(inputs), targets)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
    expected_trace = sum(g.pow(2).sum() for g in grads if g is not None)

    assert torch.allclose(efim_trace, expected_trace)


def test_empirical_fisher_trace_cross_entropy():
    """Test EFIM trace computation with CrossEntropy loss."""
    model = ClassificationModel()
    inputs = torch.randn(5, 10)
    targets = torch.randint(0, 3, (5,))

    efim_trace = empirical_fisher_trace(model, F.cross_entropy, inputs, targets)

    assert isinstance(efim_trace, torch.Tensor)
    assert efim_trace.ndim == 0

    loss = F.cross_entropy(model(inputs), targets)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
    expected_trace = sum(g.pow(2).sum() for g in grads if g is not None)

    assert torch.allclose(efim_trace, expected_trace)


def test_empirical_fisher_trace_num_samples_ignored():
    """Test that num_samples is effectively ignored for EFIM trace as it's exact."""
    model = SimpleModel()
    inputs = torch.randn(5, 10)
    targets = torch.randn(5, 1)

    trace_1_sample = empirical_fisher_trace(
        model, nn.MSELoss(), inputs, targets, num_samples=1
    )
    trace_100_samples = empirical_fisher_trace(
        model, nn.MSELoss(), inputs, targets, num_samples=100
    )

    # Since EFIM trace is exact by summing squared gradients, num_samples shouldn't change result
    assert torch.allclose(trace_1_sample, trace_100_samples)


def test_generalized_fisher_trace_nll():
    model = ClassificationModel()
    inputs = torch.randn(10, 10)  # 10 samples, 10 features
    targets = torch.randint(0, 3, (10,))  # 3 classes

    # Get outputs from the model
    outputs = model(inputs)

    # Expected behavior: GFIM trace for NLL is sum of squared gradients of negative log-likelihood
    log_likelihood = -F.nll_loss(
        F.log_softmax(outputs, dim=-1), targets, reduction="sum"
    )
    grads = torch.autograd.grad(
        log_likelihood, model.parameters(), create_graph=False, retain_graph=True
    )
    expected_trace = torch.sum(
        torch.stack([g.pow(2).sum() for g in grads if g is not None])
    )

    trace = generalized_fisher_trace(model, outputs, targets, loss_type="nll")
    assert torch.isclose(trace, expected_trace, atol=1e-5)


def test_generalized_fisher_trace_value_error():
    model = ClassificationModel()
    outputs = torch.randn(10, 5, requires_grad=True)
    targets = torch.randint(0, 5, (10,))
    with pytest.raises(ValueError):
        generalized_fisher_trace(model, outputs, targets, num_samples=0)


def test_generalized_fisher_trace_not_implemented_error_loss_type():
    model = ClassificationModel()
    outputs = torch.randn(10, 5, requires_grad=True)
    targets = torch.randint(0, 5, (10,))
    with pytest.raises(NotImplementedError):
        generalized_fisher_trace(model, outputs, targets, loss_type="unsupported")


def test_generalized_fisher_trace_not_implemented_error_shape():
    model = ClassificationModel()
    outputs = torch.randn(10, requires_grad=True)  # Incorrect shape
    targets = torch.randint(0, 5, (10,))
    with pytest.raises(NotImplementedError):
        generalized_fisher_trace(model, outputs, targets, loss_type="nll")
