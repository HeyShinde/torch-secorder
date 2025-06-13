Eigensolvers
============

.. currentmodule:: torch_secorder.analysis.eigensolvers

This module provides iterative methods for computing eigenvalues and eigenvectors of large matrices that can
only be accessed through matrix-vector products. These are particularly useful for analyzing the curvature
of neural networks.

Power Iteration
---------------

.. autofunction:: power_iteration

Example:

.. code-block:: python

    import torch
    from torch_secorder.analysis.eigensolvers import power_iteration

    # Create a simple symmetric matrix
    A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])

    def matrix_vector_product(v):
        return A @ v

    # Compute top-2 eigenvalues and eigenvectors
    eigenvalues, eigenvectors = power_iteration(
        matrix_vector_product,
        dim=2,
        num_iterations=100,
        num_vectors=2,
        tol=1e-6
    )

    print(f"Eigenvalues: {eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}")

Lanczos Algorithm
-----------------

.. autofunction:: lanczos

Example:

.. code-block:: python

    import torch
    from torch_secorder.analysis.eigensolvers import lanczos

    # Create a simple symmetric matrix
    A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])

    def matrix_vector_product(v):
        return A @ v

    # Compute top-2 eigenvalues and eigenvectors
    eigenvalues, eigenvectors = lanczos(
        matrix_vector_product,
        dim=2,
        num_iterations=100,
        num_vectors=2,
        tol=1e-6
    )

    print(f"Eigenvalues: {eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}")

Model Eigenvalues
-----------------

.. autofunction:: model_eigenvalues

Example:

.. code-block:: python

    import torch
    import torch.nn as nn
    from torch_secorder.analysis.eigensolvers import model_eigenvalues

    # Create a simple linear model
    model = nn.Linear(2, 1)

    # Create dummy data
    data = torch.randn(10, 2)
    target = torch.randn(10, 1)

    def loss_fn(model, data, target):
        return nn.functional.mse_loss(model(data), target)

    # Compute top eigenvalue and eigenvector using Lanczos
    eigenvalues, eigenvectors = model_eigenvalues(
        model,
        loss_fn,
        data,
        target,
        num_eigenvalues=1,
        method="lanczos",
        num_iterations=100,
        tol=1e-6
    )

    print(f"Top eigenvalue: {eigenvalues[0]}")
    print("Top eigenvector (in parameter space):")
    for param, eigenvector in zip(model.parameters(), eigenvectors[0]):
        print(f"Parameter shape: {param.shape}")
        print(f"Eigenvector shape: {eigenvector.shape}")
        print(f"Eigenvector values:\n{eigenvector}\n")

Notes
-----

1.  Both Power Iteration and Lanczos methods are iterative algorithms suitable for large, sparse, or implicit
    matrices where explicit matrix formation is infeasible.
2.  `model_eigenvalues` wraps these general eigensolvers to specifically compute the eigenvalues and eigenvectors
    of the Hessian (or Fisher) matrix of a neural network with respect to a given loss function, by generating
    Hessian-vector products (HVPs) or Fisher-vector products (FVPs).
