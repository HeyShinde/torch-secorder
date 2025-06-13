"""Tests for loss landscape visualization functions."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_secorder.analysis.landscape import (
    compute_loss_surface_1d,
    compute_loss_surface_2d,
    create_random_direction,
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def model():
    # A simple model for testing
    m = SimpleModel()
    # Set fixed parameters for reproducibility
    with torch.no_grad():
        m.linear.weight.fill_(0.5)
        m.linear.bias.fill_(1.0)
    return m


@pytest.fixture
def inputs():
    return torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)


@pytest.fixture
def targets():
    return torch.tensor([[5.0], [10.0]], dtype=torch.float32)


def loss_fn(outputs, targets):
    return F.mse_loss(outputs, targets)


def test_create_random_direction(model):
    direction = create_random_direction(model)
    assert isinstance(direction, list)
    assert len(direction) == len(list(model.parameters()))

    # Check that directions have the same shape as parameters and are normalized
    for d, p in zip(direction, model.parameters()):
        assert d.shape == p.shape
        assert torch.isclose(
            torch.norm(d), torch.tensor(1.0), atol=1e-6
        ) or torch.allclose(d, torch.zeros_like(d))


def test_compute_loss_surface_1d(model, inputs, targets):
    direction = create_random_direction(model)
    alphas, losses = compute_loss_surface_1d(model, loss_fn, inputs, targets, direction)

    assert alphas.shape == (50,)
    assert losses.shape == (50,)
    assert torch.all(losses >= 0)  # Loss should be non-negative

    # Verify original parameters are restored
    for p in model.parameters():
        if p.numel() == 2:  # weight
            assert torch.allclose(p.data, torch.tensor([[0.5, 0.5]]))
        elif p.numel() == 1:  # bias
            assert torch.allclose(p.data, torch.tensor([1.0]))


def test_compute_loss_surface_2d(model, inputs, targets):
    direction1 = create_random_direction(model)
    direction2 = create_random_direction(model)

    # Ensure directions are not exactly the same (though random generation makes this unlikely)
    # Or, for a more robust test, use orthogonal directions if needed

    alphas, betas, losses_surface = compute_loss_surface_2d(
        model, loss_fn, inputs, targets, direction1, direction2
    )

    assert alphas.shape == (25,)
    assert betas.shape == (25,)
    assert losses_surface.shape == (25, 25)
    assert torch.all(losses_surface >= 0)  # Loss should be non-negative

    # Verify original parameters are restored
    for p in model.parameters():
        if p.numel() == 2:  # weight
            assert torch.allclose(p.data, torch.tensor([[0.5, 0.5]]))
        elif p.numel() == 1:  # bias
            assert torch.allclose(p.data, torch.tensor([1.0]))
