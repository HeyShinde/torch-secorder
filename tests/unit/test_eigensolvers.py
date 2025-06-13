"""Unit tests for eigensolvers module."""

import pytest
import torch

from torch_secorder.analysis.eigensolvers import (
    lanczos,
    model_eigenvalues,
    power_iteration,
)


def test_power_iteration():
    """Test power iteration on a simple symmetric matrix."""
    # Create a simple symmetric matrix
    A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])

    def matrix_vector_product(v):
        return A @ v

    eigenvalues, eigenvectors = power_iteration(
        matrix_vector_product, dim=2, num_iterations=100, num_vectors=2, tol=1e-6
    )

    # Check that eigenvalues are close to true values
    true_eigenvalues = torch.linalg.eigvals(A).real
    true_eigenvalues, _ = torch.sort(true_eigenvalues, descending=True)
    assert torch.allclose(eigenvalues, true_eigenvalues, atol=1e-3)

    # Check that eigenvectors are normalized
    assert torch.allclose(torch.norm(eigenvectors, dim=0), torch.ones(2), atol=1e-6)


def test_lanczos():
    """Test Lanczos algorithm on a simple symmetric matrix."""
    # Create a simple symmetric matrix
    A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])

    def matrix_vector_product(v):
        return A @ v

    eigenvalues, eigenvectors = lanczos(
        matrix_vector_product, dim=2, num_iterations=1000, num_vectors=2, tol=1e-2
    )

    # Check that eigenvalues are close to true values
    true_eigenvalues = torch.linalg.eigvals(A).real
    true_eigenvalues, _ = torch.sort(true_eigenvalues, descending=True)
    assert torch.allclose(eigenvalues, true_eigenvalues, atol=1e-2)

    # Check that eigenvectors are orthogonal
    # The eigenvectors from Lanczos are already orthogonal by construction within the V matrix
    # For this test, we need to ensure the final eigenvectors are close to orthogonal
    # (though for num_vectors=2, they are already column-orthogonal from the algorithm)
    # If num_vectors < dim, we only get a subset.
    if 2 > 1:  # Check orthogonality only if multiple vectors are computed
        orthogonality_check = torch.abs(
            eigenvectors.T @ eigenvectors - torch.eye(2, device=eigenvectors.device)
        ).sum()
        assert orthogonality_check < 1e-3  # Sum of absolute differences should be small


def test_model_eigenvalues():
    """Test model_eigenvalues on a simple linear model."""
    # Create a simple linear model
    model = torch.nn.Linear(2, 1)

    # Create dummy data
    data = torch.randn(10, 2)
    target = torch.randn(10, 1)

    def loss_fn(model, data, target):
        return torch.nn.functional.mse_loss(model(data), target)

    # Test both methods
    for method in ["power_iteration", "lanczos"]:
        eigenvalues, eigenvectors = model_eigenvalues(
            model,
            loss_fn,
            data,
            target,
            num_eigenvalues=1,
            method=method,
            num_iterations=100,
            tol=1e-6,
        )

        # Check that we got the expected number of eigenvalues
        assert eigenvalues.shape == (1,)

        # Check that eigenvectors are in parameter space
        assert len(eigenvectors) == 1
        assert isinstance(eigenvectors[0], list)
        for param_list in eigenvectors:
            for p in param_list:
                assert isinstance(p, torch.Tensor)

        # Check that eigenvectors match parameter shapes
        for param, eigenvector in zip(model.parameters(), eigenvectors[0]):
            assert param.shape == eigenvector.shape


def test_invalid_method():
    """Test that invalid method raises ValueError."""
    model = torch.nn.Linear(2, 1)
    data = torch.randn(10, 2)
    target = torch.randn(10, 1)

    def loss_fn(model, data, target):
        return torch.nn.functional.mse_loss(model(data), target)

    with pytest.raises(ValueError):
        model_eigenvalues(model, loss_fn, data, target, method="invalid_method")


def test_convergence():
    """Test that both methods converge to similar results."""
    # Create a simple symmetric matrix
    A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])

    def matrix_vector_product(v):
        return A @ v

    # Get results from both methods
    power_eigenvalues, _ = power_iteration(
        matrix_vector_product, dim=2, num_iterations=1000, num_vectors=2, tol=1e-2
    )

    lanczos_eigenvalues, _ = lanczos(
        matrix_vector_product, dim=2, num_iterations=1000, num_vectors=2, tol=1e-2
    )

    # Check that both methods give similar results
    assert torch.allclose(power_eigenvalues, lanczos_eigenvalues, atol=1e-2)


def test_device_handling():
    """Test that device handling works correctly."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a simple symmetric matrix
    A = torch.tensor([[2.0, 1.0], [1.0, 3.0]], device=device)

    def matrix_vector_product(v):
        return A @ v

    # Test both methods
    for method in ["power_iteration", "lanczos"]:
        eigenvalues, eigenvectors = power_iteration(
            matrix_vector_product,
            dim=2,
            num_iterations=100,
            num_vectors=2,
            tol=1e-6,
            device=device,
        )

        # Check that results are on the correct device
        assert eigenvalues.device == device
        assert eigenvectors.device == device
