"""Eigensolvers for computing top-K eigenvalues and eigenvectors of Hessian/Fisher matrices.

This module provides iterative methods for computing eigenvalues and eigenvectors
of large matrices that can only be accessed through matrix-vector products.
"""

from typing import Callable, List, Optional, Tuple

import torch

from ..core.utils import flatten_params, unflatten_params


def power_iteration(
    matrix_vector_product: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    num_iterations: int = 100,
    num_vectors: int = 1,
    tol: float = 1e-6,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the top-K eigenvalues and eigenvectors using power iteration.

    Args:
        matrix_vector_product: Function that computes matrix-vector product
        dim: Dimension of the matrix
        num_iterations: Maximum number of iterations
        num_vectors: Number of top eigenvectors to compute
        tol: Convergence tolerance
        device: Device to use for computation

    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize random vectors
    vectors = torch.randn(dim, num_vectors, device=device)
    vectors = vectors / torch.norm(vectors, dim=0, keepdim=True)

    eigenvalues = torch.zeros(num_vectors, device=device)
    prev_eigenvalues = torch.zeros(num_vectors, device=device)

    for _ in range(num_iterations):
        # Matrix-vector products
        vectors = matrix_vector_product(vectors)

        # Orthogonalize using Gram-Schmidt
        for i in range(num_vectors):
            for j in range(i):
                vectors[:, i] -= torch.dot(vectors[:, i], vectors[:, j]) * vectors[:, j]
            vectors[:, i] = vectors[:, i] / torch.norm(vectors[:, i])

        # Compute eigenvalues
        eigenvalues = torch.diag(vectors.T @ matrix_vector_product(vectors))

        # Check convergence
        if torch.all(torch.abs(eigenvalues - prev_eigenvalues) < tol):
            break

        prev_eigenvalues = eigenvalues.clone()

    return eigenvalues, vectors


def lanczos(
    matrix_vector_product: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    num_iterations: int = 100,
    num_vectors: int = 1,
    tol: float = 1e-6,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the top-K eigenvalues and eigenvectors using Lanczos algorithm.

    Args:
        matrix_vector_product: Function that computes matrix-vector product
        dim: Dimension of the matrix
        num_iterations: Maximum number of iterations
        num_vectors: Number of top eigenvectors to compute
        tol: Convergence tolerance
        device: Device to use for computation

    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize first vector
    v = torch.randn(dim, device=device)
    v = v / torch.norm(v)

    # Initialize Lanczos vectors and tridiagonal matrix
    V = torch.zeros(dim, num_iterations, device=device)
    T = torch.zeros(num_iterations, num_iterations, device=device)
    V[:, 0] = v

    # First iteration
    w = matrix_vector_product(v)
    alpha = torch.dot(w, v)
    T[0, 0] = alpha
    w = w - alpha * v
    beta = torch.norm(w)
    T[0, 1] = beta
    T[1, 0] = beta

    # Main Lanczos iterations
    for i in range(1, num_iterations):
        if beta < tol:
            break

        v = w / beta
        V[:, i] = v

        w = matrix_vector_product(v)
        w = w - beta * V[:, i - 1]
        alpha = torch.dot(w, v)
        T[i, i] = alpha
        w = w - alpha * v
        beta = torch.norm(w)

        if i < num_iterations - 1:
            T[i, i + 1] = beta
            T[i + 1, i] = beta

    # Compute eigenvalues and eigenvectors of tridiagonal matrix
    eigenvals, eigenvecs = torch.linalg.eigh(T[: i + 1, : i + 1])
    eigenvals, idx = torch.sort(eigenvals, descending=True)
    eigenvecs = eigenvecs[:, idx]

    # Convert to original space
    eigenvectors = V[:, : i + 1] @ eigenvecs[:, :num_vectors]

    return eigenvals[:num_vectors], eigenvectors


def model_eigenvalues(
    model: torch.nn.Module,
    loss_fn: Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], torch.Tensor],
    data: torch.Tensor,
    target: torch.Tensor,
    num_eigenvalues: int = 1,
    method: str = "lanczos",
    num_iterations: int = 100,
    tol: float = 1e-6,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
    """Compute top-K eigenvalues and eigenvectors of the Hessian matrix for a model.

    Args:
        model: PyTorch model
        loss_fn: Loss function that takes (model, data, target) as arguments
        data: Input data
        target: Target data
        num_eigenvalues: Number of top eigenvalues to compute
        method: Method to use ('power_iteration' or 'lanczos')
        num_iterations: Maximum number of iterations
        tol: Convergence tolerance
        device: Device to use for computation

    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    if device is None:
        device = next(model.parameters()).device

    # Get flattened parameters
    params = list(model.parameters())
    param_shapes = [p.shape for p in params]
    flat_params = flatten_params(params)

    def hvp(v: torch.Tensor) -> torch.Tensor:
        """Compute Hessian-vector product. Handles both single and multiple vectors."""
        # Compute loss
        loss = loss_fn(model, data, target)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        flat_grads = flatten_params(grads)

        if v.ndim == 1:
            # Single vector
            hvp = torch.autograd.grad(
                torch.dot(flat_grads, v), params, create_graph=False, allow_unused=True
            )
            return flatten_params(hvp)
        elif v.ndim == 2:
            # Multiple vectors (batch mode)
            outs = []
            for i in range(v.shape[1]):
                retain = i != v.shape[1] - 1
                hvp = torch.autograd.grad(
                    torch.dot(flat_grads, v[:, i]),
                    params,
                    create_graph=False,
                    allow_unused=True,
                    retain_graph=retain,
                )
                outs.append(flatten_params(hvp).unsqueeze(1))
            return torch.cat(outs, dim=1)
        else:
            raise ValueError("Input vector v must be 1D or 2D tensor.")

    # Choose eigensolver
    if method.lower() == "power_iteration":
        eigenvalues, eigenvectors = power_iteration(
            hvp, flat_params.numel(), num_iterations, num_eigenvalues, tol, device
        )
    elif method.lower() == "lanczos":
        eigenvalues, eigenvectors = lanczos(
            hvp, flat_params.numel(), num_iterations, num_eigenvalues, tol, device
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Convert eigenvectors back to parameter space
    param_eigenvectors = []
    for i in range(num_eigenvalues):
        param_eigenvectors.append(unflatten_params(eigenvectors[:, i], param_shapes))

    return eigenvalues, param_eigenvectors


estimate_eigenvalues = model_eigenvalues
lanczos_iteration = lanczos

__all__ = [
    "power_iteration",
    "lanczos",
    "model_eigenvalues",
    "estimate_eigenvalues",
    "lanczos_iteration",
]
