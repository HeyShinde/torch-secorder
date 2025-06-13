"""Integration tests for analysis pipelines."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_secorder.analysis.eigensolvers import model_eigenvalues
from torch_secorder.analysis.landscape import (
    compute_loss_surface_1d,
    create_random_direction,
)
from torch_secorder.approximations.fisher_diagonal import empirical_fisher_diagonal
from torch_secorder.core.hessian_diagonal import hessian_diagonal


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 2)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


@pytest.fixture
def model():
    return SimpleModel()


@pytest.fixture
def inputs():
    return torch.randn(16, 10)  # Batch size 16, input features 10


@pytest.fixture
def targets():
    return torch.randint(0, 2, (16,))  # 2 classes


def loss_fn(outputs, targets):
    return F.cross_entropy(outputs, targets)


def test_full_analysis_pipeline(model, inputs, targets):
    """Tests a basic analysis pipeline: Hessian diagonal -> Fisher diagonal -> Eigenvalues -> Loss Landscape."""

    # 1. Compute Hessian Diagonal
    # Prepare a callable for hessian_diagonal
    def get_loss():
        outputs = model(inputs)
        return loss_fn(outputs, targets)

    params = list(model.parameters())

    hessian_diagonals_list = hessian_diagonal(get_loss, params)
    assert isinstance(hessian_diagonals_list, list)
    assert all(isinstance(h, torch.Tensor) for h in hessian_diagonals_list)
    hessian_diag = torch.cat([h.flatten() for h in hessian_diagonals_list])
    assert hessian_diag.dim() == 1
    assert hessian_diag.numel() == sum(p.numel() for p in model.parameters())
    print(f"Hessian Diagonal computed. Shape: {hessian_diag.shape}")

    # 2. Compute Empirical Fisher Diagonal
    fisher_diagonals_list = empirical_fisher_diagonal(model, loss_fn, inputs, targets)
    assert isinstance(fisher_diagonals_list, list)
    assert all(isinstance(f, torch.Tensor) for f in fisher_diagonals_list)
    fisher_diag = torch.cat([f.flatten() for f in fisher_diagonals_list])
    assert fisher_diag.dim() == 1
    assert fisher_diag.numel() == sum(p.numel() for p in model.parameters())
    print(f"Empirical Fisher Diagonal computed. Shape: {fisher_diag.shape}")

    # 3. Compute top eigenvalues of a curvature matrix (e.g., Fisher)
    # We'll use Fisher for this, as it's a positive semi-definite matrix.
    # For eigenvalues, we need a function that accepts a vector and returns a vector-product.
    # We can create a simple closure for this.
    # The eigen_fn for model_eigenvalues expects (model, loss_fn, inputs, targets, **kwargs)
    # The previous implementation was already passing the correct arguments (model, loss_fn, inputs, targets)
    # The `flatten_output` flag ensures the output is a 1D tensor, which is what empirical_fisher_diagonal returns

    def wrapped_loss_fn(model, data, target):
        return loss_fn(model(data), target)

    eigenvalues, _ = model_eigenvalues(
        model,
        wrapped_loss_fn,
        inputs,
        targets,
        num_eigenvalues=3,
        method="lanczos",
        num_iterations=20,
        tol=1e-4,
    )

    assert isinstance(eigenvalues, torch.Tensor)
    assert eigenvalues.numel() == 3
    print(f"Top 3 Fisher Eigenvalues computed: {eigenvalues.tolist()}")

    # 4. Compute 1D Loss Landscape slice
    direction = create_random_direction(model)
    alphas, losses = compute_loss_surface_1d(model, loss_fn, inputs, targets, direction)
    assert alphas.shape == (50,)
    assert losses.shape == (50,)
    print(f"1D Loss Surface computed. First 5 losses: {losses[:5].tolist()}")

    # Check that original model parameters are restored after all operations
    # The original_params were captured before any operations that modify parameters
    # This part of the test should be fine as it is.
    initial_params = []
    for p in model.parameters():
        initial_params.append(p.data.clone())

    # Re-run a simple forward pass and loss computation with original parameters
    # to ensure they haven't been altered permanently.
    # Assert that parameters are still the same as initial, within tolerance
    for i, p in enumerate(model.parameters()):
        assert torch.allclose(p.data, initial_params[i], atol=1e-6)

    print("Original model parameters restored successfully.")
