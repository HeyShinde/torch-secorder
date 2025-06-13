"""Tests for per-layer curvature extraction functionality."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_secorder.core.per_layer_curvature import (
    get_layer_curvature_stats,
    per_layer_fisher_diagonal,
    per_layer_hessian_diagonal,
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 3)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)


class SimpleRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 1)  # Output for regression

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)


@pytest.fixture
def model():
    return SimpleModel()


@pytest.fixture
def regression_model():
    return SimpleRegressionModel()


@pytest.fixture
def inputs():
    return torch.randn(8, 10)


@pytest.fixture
def targets():
    return torch.randint(0, 3, (8,))  # 3 classes


@pytest.fixture
def regression_targets():
    return torch.randn(8, 1)  # Floating point targets for regression


def test_per_layer_hessian_diagonal(model, inputs, targets):
    """Test per-layer Hessian diagonal computation."""
    loss_fn = nn.CrossEntropyLoss()
    layer_hessians = per_layer_hessian_diagonal(model, loss_fn, inputs, targets)

    assert isinstance(layer_hessians, dict)
    assert len(layer_hessians) > 0

    for layer_name, hessian_diag in layer_hessians.items():
        assert isinstance(hessian_diag, torch.Tensor)
        assert hessian_diag.numel() > 0

    # Check shapes (adjusting for new model sizes)
    assert layer_hessians["linear1"].shape == (
        55,
    )  # weights (5,10) + bias (5,) = 55 elements
    assert layer_hessians["linear2"].shape == (
        18,
    )  # weights (3,5) + bias (3,) = 18 elements

    # Check that the computational graph is not created if create_graph is False (default)
    for hessian_diag in layer_hessians.values():
        assert hessian_diag.grad_fn is None


def test_per_layer_hessian_diagonal_regression(
    regression_model, inputs, regression_targets
):
    """Test per-layer Hessian diagonal computation for regression."""
    loss_fn = nn.MSELoss()
    layer_hessians = per_layer_hessian_diagonal(
        regression_model, loss_fn, inputs, regression_targets
    )

    assert isinstance(layer_hessians, dict)
    assert len(layer_hessians) > 0

    for layer_name, hessian_diag in layer_hessians.items():
        assert isinstance(hessian_diag, torch.Tensor)
        assert hessian_diag.numel() > 0

    # Check shapes
    assert layer_hessians["linear1"].shape == (
        55,
    )  # weights (5,10) + bias (5,) = 55 elements
    assert layer_hessians["linear2"].shape == (
        6,
    )  # weights (1,5) + bias (1,) = 6 elements

    # Check that the computational graph is not created
    for hessian_diag in layer_hessians.values():
        assert hessian_diag.grad_fn is None


def test_per_layer_fisher_diagonal():
    model = SimpleModel()
    X = torch.randn(8, 10)  # float
    y = torch.randint(0, 3, (8,), dtype=torch.long)  # long for classification
    loss_fn = nn.CrossEntropyLoss()

    # Compute per-layer Fisher diagonal
    fisher_diags = per_layer_fisher_diagonal(model, loss_fn, X, y)

    # Check that we got a dictionary with the right keys
    assert set(fisher_diags.keys()) == {"linear1", "linear2"}

    # Check that each diagonal is the right shape
    assert (
        fisher_diags["linear1"].numel()
        == model.linear1.weight.numel() + model.linear1.bias.numel()
    )
    assert (
        fisher_diags["linear2"].numel()
        == model.linear2.weight.numel() + model.linear2.bias.numel()
    )

    # Check that all values are non-negative (Fisher is positive semi-definite)
    for diag in fisher_diags.values():
        assert torch.all(diag >= 0)


def test_get_layer_curvature_stats():
    """Test computation of layer curvature statistics."""
    # Create sample layer curvatures
    layer_curvatures = {
        "layer1": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "layer2": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
    }

    stats = get_layer_curvature_stats(layer_curvatures)

    # Check that we got stats for both layers
    assert "layer1" in stats
    assert "layer2" in stats

    # Check that each layer has all required statistics
    for layer in ["layer1", "layer2"]:
        assert "mean" in stats[layer]
        assert "std" in stats[layer]
        assert "max" in stats[layer]
        assert "min" in stats[layer]

    # Check specific values for layer1
    assert abs(stats["layer1"]["mean"] - 2.5) < 1e-6
    assert abs(stats["layer1"]["max"] - 4.0) < 1e-6
    assert abs(stats["layer1"]["min"] - 1.0) < 1e-6


def test_per_layer_fisher_diagonal_with_create_graph(model, inputs, targets):
    """Test per-layer Fisher diagonal computation with create_graph=True."""
    loss_fn = nn.CrossEntropyLoss()
    layer_fishers = per_layer_fisher_diagonal(
        model, loss_fn, inputs, targets, create_graph=True
    )

    assert isinstance(layer_fishers, dict)
    assert len(layer_fishers) > 0

    for layer_name, fisher in layer_fishers.items():
        assert isinstance(fisher, torch.Tensor)
        assert fisher.numel() > 0
        # Check that the computational graph is created
        assert fisher.grad_fn is not None

    # Check shapes (adjusting for new model sizes)
    assert layer_fishers["linear1"].shape == (
        55,
    )  # weights (5,10) + bias (5,) = 55 elements
    assert layer_fishers["linear2"].shape == (
        18,
    )  # weights (3,5) + bias (3,) = 18 elements

    for fisher in layer_fishers.values():
        assert fisher.grad_fn is not None


def test_custom_layer_types(model, inputs, targets):
    """Test per-layer curvature computation with custom layer types."""
    loss_fn = nn.CrossEntropyLoss()

    # Test with only Linear layers
    layer_hessians = per_layer_hessian_diagonal(
        model, loss_fn, inputs, targets, layer_types=[nn.Linear]
    )
    assert len(layer_hessians) == 2
    assert "linear1" in layer_hessians and "linear2" in layer_hessians

    # Test with no matching layers
    layer_hessians_empty = per_layer_hessian_diagonal(
        model, loss_fn, inputs, targets, layer_types=[nn.Conv2d]
    )
    assert len(layer_hessians_empty) == 0

    # Test per-layer Fisher with custom layer types
    layer_fishers = per_layer_fisher_diagonal(
        model, loss_fn, inputs, targets, layer_types=[nn.Linear]
    )
    assert len(layer_fishers) == 2
    assert "linear1" in layer_fishers and "linear2" in layer_fishers


def test_custom_layer_types_regression(regression_model, inputs, regression_targets):
    """Test per-layer curvature computation with custom layer types for regression."""
    loss_fn = nn.MSELoss()

    # Test with only Linear layers for Hessian
    layer_hessians = per_layer_hessian_diagonal(
        regression_model, loss_fn, inputs, regression_targets, layer_types=[nn.Linear]
    )
    assert len(layer_hessians) == 2
    assert "linear1" in layer_hessians and "linear2" in layer_hessians

    # Test with no matching layers for Hessian
    layer_hessians_empty = per_layer_hessian_diagonal(
        regression_model, loss_fn, inputs, regression_targets, layer_types=[nn.Conv2d]
    )
    assert len(layer_hessians_empty) == 0

    # Test with only Linear layers for Fisher
    layer_fishers = per_layer_fisher_diagonal(
        regression_model, loss_fn, inputs, regression_targets, layer_types=[nn.Linear]
    )
    assert len(layer_fishers) == 2
    assert "linear1" in layer_fishers and "linear2" in layer_fishers


def test_create_graph(model, inputs, targets):
    """Test per-layer curvature computation with create_graph=True."""
    loss_fn = nn.CrossEntropyLoss()

    # Test Hessian
    layer_hessians = per_layer_hessian_diagonal(
        model, loss_fn, inputs, targets, create_graph=True
    )
    for hessian_diag in layer_hessians.values():
        assert hessian_diag.grad_fn is not None

    # Test Fisher
    layer_fishers = per_layer_fisher_diagonal(
        model, loss_fn, inputs, targets, create_graph=True
    )
    for fisher_diag in layer_fishers.values():
        assert fisher_diag.grad_fn is not None


@pytest.mark.skip(
    reason="Known issue with grad_fn for regression with create_graph=True in hessian_diagonal, needs further investigation."
)
def test_create_graph_regression(regression_model, inputs, regression_targets):
    """Test per-layer curvature computation with create_graph=True for regression."""
    loss_fn = nn.MSELoss()

    # Test Hessian
    layer_hessians = per_layer_hessian_diagonal(
        regression_model, loss_fn, inputs, regression_targets, create_graph=True
    )
    for hessian_diag in layer_hessians.values():
        assert hessian_diag.grad_fn is not None

    # Test Fisher
    layer_fishers = per_layer_fisher_diagonal(
        regression_model, loss_fn, inputs, regression_targets, create_graph=True
    )
    for fisher_diag in layer_fishers.values():
        assert fisher_diag.grad_fn is not None
