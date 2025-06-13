Gauss-Newton Module
===================

.. currentmodule:: torch_secorder.approximations.gauss_newton

The Gauss-Newton module provides functions for computing approximations related to the Gauss-Newton Matrix.

Gauss-Newton Matrix Approximation
---------------------------------

.. autofunction:: gauss_newton_matrix_approximation

Example:
~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   from torch_secorder.approximations.gauss_newton import gauss_newton_matrix_approximation

   # Define a simple linear model
   class SimpleModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.linear = nn.Linear(10, 1)

       def forward(self, x):
           return self.linear(x)

   model = SimpleModel()
   loss_fn = nn.MSELoss()
   inputs = torch.randn(5, 10)
   targets = torch.randn(5, 1)

   # Compute the diagonal of the Gauss-Newton Matrix approximation
   gnm_diagonal = gauss_newton_matrix_approximation(model, loss_fn, inputs, targets)

   print("Gauss-Newton Matrix Diagonal (for weights):", gnm_diagonal[0].shape)
   print("Gauss-Newton Matrix Diagonal (for bias):", gnm_diagonal[1].shape)

Notes
-----

1. The Gauss-Newton Matrix is a positive semi-definite approximation of the Hessian, commonly used in non-linear least squares optimization.

2. This function provides the diagonal elements of the GNM, which can be used for diagonal approximations in optimizers or for diagnostic purposes.

3. The `create_graph` parameter allows for computing higher-order derivatives, which is useful in meta-learning or other advanced scenarios.
