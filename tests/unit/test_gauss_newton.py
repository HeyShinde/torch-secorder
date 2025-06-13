"""Tests for Gauss-Newton matrix approximation functions."""

import pytest
import torch
import torch.nn as nn

from torch_secorder.approximations.gauss_newton import gauss_newton_matrix_approximation


class SimpleLinearModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


def test_gauss_newton_matrix_approximation_linear_mse():
    """Test GNM diagonal for a simple linear model with MSE loss."""
    model = SimpleLinearModel(10, 1)
    inputs = torch.randn(5, 10)
    targets = torch.randn(5, 1)
    loss_fn = nn.MSELoss()

    gnm_diag = gauss_newton_matrix_approximation(model, loss_fn, inputs, targets)

    assert len(gnm_diag) == 2  # weight and bias
    assert gnm_diag[0].shape == model.linear.weight.shape
    assert gnm_diag[1].shape == model.linear.bias.shape

    # Manual computation of GNM diagonal for verification
    # For L = 1/2 ||f(x;theta) - y||^2, GNM = J^T J where J = d f / d theta
    # The diagonal of GNM is sum_{k} (d f_k / d theta_i)^2

    # Compute Jacobian of model outputs with respect to parameters
    outputs = model(inputs)

    # Initialize list to hold squared gradients summed across output dimensions
    expected_gnm_diag_weight = torch.zeros_like(model.linear.weight)
    expected_gnm_diag_bias = torch.zeros_like(model.linear.bias)

    for i in range(outputs.numel()):
        output_element = outputs.view(-1)[i]
        grads = torch.autograd.grad(
            output_element, model.parameters(), retain_graph=True
        )

        if grads[0] is not None:
            expected_gnm_diag_weight += grads[0].pow(2)
        if grads[1] is not None:
            expected_gnm_diag_bias += grads[1].pow(2)

    assert torch.allclose(gnm_diag[0], expected_gnm_diag_weight, atol=1e-5)
    assert torch.allclose(gnm_diag[1], expected_gnm_diag_bias, atol=1e-5)


def test_gauss_newton_matrix_approximation_value_error():
    """Test that ValueError is raised for non-MSE loss functions."""
    model = SimpleLinearModel(5, 1)
    inputs = torch.randn(2, 5)
    targets = torch.randn(2, 1)
    loss_fn = nn.L1Loss()  # Non-MSE loss

    with pytest.raises(
        ValueError,
        match="Gauss-Newton Matrix approximation is typically used with MSE-based loss functions",
    ):
        gauss_newton_matrix_approximation(model, loss_fn, inputs, targets)


def test_gauss_newton_matrix_approximation_no_grad_params():
    """Test GNM diagonal when some parameters don't require gradients."""

    class ModelWithNoGrad(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(5, 5)
            self.linear2 = nn.Linear(5, 1)
            self.linear1.weight.requires_grad_(False)
            self.linear1.bias.requires_grad_(False)

        def forward(self, x):
            x = self.linear1(x)
            return self.linear2(x)

    model = ModelWithNoGrad()
    inputs = torch.randn(2, 5)
    targets = torch.randn(2, 1)
    loss_fn = nn.MSELoss()

    gnm_diag = gauss_newton_matrix_approximation(model, loss_fn, inputs, targets)

    # Only linear2's weight and bias should have GNM diagonal components
    # The implementation implicitly handles None gradients by not adding to gnm_diagonal
    # So, we should expect 2 elements in the list if linear1 params were filtered out
    # Current implementation returns based on non-None grads from `torch.autograd.grad`
    # If no_grad params produce None grads, they won't be in the result list.

    # Let's test the shapes and that the elements correspond to linear2
    assert len(gnm_diag) == 2  # linear2.weight and linear2.bias
    assert gnm_diag[0].shape == model.linear2.weight.shape
    assert gnm_diag[1].shape == model.linear2.bias.shape

    # Manually verify linear2's GNM diagonal
    outputs = model(inputs)
    expected_gnm_diag_weight_l2 = torch.zeros_like(model.linear2.weight)
    expected_gnm_diag_bias_l2 = torch.zeros_like(model.linear2.bias)

    for i in range(outputs.numel()):
        output_element = outputs.view(-1)[i]
        # Only get gradients for linear2 parameters
        grads = torch.autograd.grad(
            output_element,
            [model.linear2.weight, model.linear2.bias],
            retain_graph=True,
        )

        if grads[0] is not None:
            expected_gnm_diag_weight_l2 += grads[0].pow(2)
        if grads[1] is not None:
            expected_gnm_diag_bias_l2 += grads[1].pow(2)

    assert torch.allclose(gnm_diag[0], expected_gnm_diag_weight_l2, atol=1e-5)
    assert torch.allclose(gnm_diag[1], expected_gnm_diag_bias_l2, atol=1e-5)


def test_gauss_newton_matrix_approximation_create_graph():
    """Test GNM diagonal with create_graph=True."""
    model = SimpleLinearModel(5, 1)
    inputs = torch.randn(2, 5)
    targets = torch.randn(2, 1)
    loss_fn = nn.MSELoss()

    gnm_diag = gauss_newton_matrix_approximation(
        model, loss_fn, inputs, targets, create_graph=True
    )

    # Check that the returned tensors have a grad_fn, indicating graph was created
    assert gnm_diag[0].grad_fn is not None
    assert gnm_diag[1].grad_fn is not None

    # You could perform a second backward pass if needed for higher-order derivatives
    # (e.g., in meta-learning scenarios)
    # For this test, just checking grad_fn is sufficient to confirm graph creation.
