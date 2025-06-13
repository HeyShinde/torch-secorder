"""Tests for the Fisher diagonal computation functions."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_secorder.approximations.fisher_diagonal import (
    empirical_fisher_diagonal,
    generalized_fisher_diagonal,
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


def test_empirical_fisher_diagonal_mse():
    """Test EFIM diagonal computation with MSE loss."""
    model = SimpleModel()
    inputs = torch.randn(5, 10)
    targets = torch.randn(5, 1)

    efim_diag = empirical_fisher_diagonal(model, nn.MSELoss(), inputs, targets)

    assert len(efim_diag) == 2  # weight and bias
    assert efim_diag[0].shape == model.linear.weight.shape
    assert efim_diag[1].shape == model.linear.bias.shape

    # Manual computation of squared gradients for verification
    loss = nn.MSELoss()(model(inputs), targets)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    expected_diag_weight = grads[0].pow(2)
    expected_diag_bias = grads[1].pow(2)

    assert torch.allclose(efim_diag[0], expected_diag_weight)
    assert torch.allclose(efim_diag[1], expected_diag_bias)


def test_empirical_fisher_diagonal_cross_entropy():
    """Test EFIM diagonal computation with CrossEntropy loss."""
    model = ClassificationModel()
    inputs = torch.randn(5, 10)
    targets = torch.randint(0, 3, (5,))

    efim_diag = empirical_fisher_diagonal(model, F.cross_entropy, inputs, targets)

    assert len(efim_diag) == 2  # weight and bias
    assert efim_diag[0].shape == model.linear.weight.shape
    assert efim_diag[1].shape == model.linear.bias.shape

    # Manual computation of squared gradients for verification
    loss = F.cross_entropy(model(inputs), targets)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    expected_diag_weight = grads[0].pow(2)
    expected_diag_bias = grads[1].pow(2)

    assert torch.allclose(efim_diag[0], expected_diag_weight)
    assert torch.allclose(efim_diag[1], expected_diag_bias)


def test_generalized_fisher_diagonal_nll():
    model = ClassificationModel()
    inputs = torch.randn(10, 10)  # 10 samples, 10 features
    targets = torch.randint(0, 3, (10,))  # 3 classes

    # Get outputs from the model
    outputs = model(inputs)

    # Expected behavior: GFIM diagonal for NLL is squared gradients of negative log-likelihood
    log_likelihood = -F.nll_loss(
        F.log_softmax(outputs, dim=-1), targets, reduction="sum"
    )
    grads = torch.autograd.grad(
        log_likelihood, model.parameters(), create_graph=False, retain_graph=True
    )
    expected_diag = [g.pow(2) for g in grads if g is not None]

    diag = generalized_fisher_diagonal(model, outputs, targets, loss_type="nll")
    assert len(diag) == len(expected_diag)
    for d, e in zip(diag, expected_diag):
        assert torch.allclose(d, e, atol=1e-5)


def test_generalized_fisher_diagonal_not_implemented_error_loss_type():
    model = ClassificationModel()
    inputs = torch.randn(10, 10)
    targets = torch.randint(0, 3, (10,))
    outputs = model(inputs)
    with pytest.raises(NotImplementedError):
        generalized_fisher_diagonal(model, outputs, targets, loss_type="unsupported")


def test_generalized_fisher_diagonal_not_implemented_error_shape():
    model = ClassificationModel()
    outputs = torch.randn(10, requires_grad=True)  # Incorrect shape
    targets = torch.randint(0, 3, (10,))
    with pytest.raises(NotImplementedError):
        generalized_fisher_diagonal(model, outputs, targets, loss_type="nll")
