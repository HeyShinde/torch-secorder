Hessian Diagonal Module
=======================

.. currentmodule:: torch_secorder.core.hessian_diagonal

The Hessian Diagonal module provides functions to compute the diagonal elements of the Hessian matrix.

Hessian Diagonal
----------------

.. autofunction:: hessian_diagonal

Computes the diagonal elements of the Hessian matrix of a scalar `output` with respect to `inputs`.

Example:
~~~~~~~~

.. code-block:: python

    import torch
    from torch_secorder.core.hessian_diagonal import hessian_diagonal

    # Create a simple quadratic function: f(x) = x^T A x
    A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
    x = torch.tensor([1.0, 2.0], requires_grad=True)

    def quadratic():
        return x @ A @ x

    # Compute the diagonal of the Hessian
    diag = hessian_diagonal(quadratic, [x])
    print(diag[0])  # Should print tensor([4., 6.]) (diagonal of 2A)

Model Hessian Diagonal
----------------------

.. autofunction:: model_hessian_diagonal

A convenience function to compute the diagonal of the Hessian of a model's loss with respect to its parameters.

Example:
~~~~~~~~

.. code-block:: python

    import torch
    import torch.nn as nn
    from torch_secorder.core.hessian_diagonal import model_hessian_diagonal

    # Create a simple neural network
    model = nn.Linear(10, 1)
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)

    # Compute the diagonal of the Hessian for the MSE loss
    diag = model_hessian_diagonal(model, nn.functional.mse_loss, x, y)

    # diag[0] contains the diagonal for the weight matrix
    # diag[1] contains the diagonal for the bias vector
    print(f"Weight diagonal shape: {diag[0].shape}")
    print(f"Bias diagonal shape: {diag[1].shape}")

Notes
-----

1. The Hessian diagonal computation uses Hessian-vector products with unit vectors to compute the diagonal elements efficiently.

2. The `create_graph` parameter allows for computing higher-order derivatives if needed, but this increases memory usage.

3. The `strict` parameter controls whether an error should be raised if any parameter requires gradients but has none.
