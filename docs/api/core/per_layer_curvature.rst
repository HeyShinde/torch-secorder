Per-Layer Curvature Module
==========================

.. currentmodule:: torch_secorder.core.per_layer_curvature

The per-layer curvature module provides utilities for computing block-diagonal approximations
of the Hessian and Fisher Information matrices, where each block corresponds to a layer's
parameters. This is useful for understanding the curvature of individual layers and for
implementing layer-wise optimization strategies.

Per-Layer Hessian Diagonal
--------------------------

.. autofunction:: per_layer_hessian_diagonal

Example:
~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   from torch_secorder.core.per_layer_curvature import per_layer_hessian_diagonal

   # Define a simple model with multiple layers
   class SimpleModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.linear1 = nn.Linear(10, 20)
           self.linear2 = nn.Linear(20, 5)
           self.linear3 = nn.Linear(5, 1)

       def forward(self, x):
           return self.linear3(self.linear2(self.linear1(x)))

   model = SimpleModel()
   loss_fn = nn.MSELoss()
   inputs = torch.randn(5, 10)
   targets = torch.randn(5, 1)

   # Compute per-layer Hessian diagonals
   layer_hessians = per_layer_hessian_diagonal(model, loss_fn, inputs, targets)

   # Print shapes of Hessian diagonals for each layer
   for layer_name, hessian in layer_hessians.items():
       print(f"{layer_name} Hessian diagonal shape: {hessian.shape}")

Per-Layer Fisher Diagonal
-------------------------

.. autofunction:: per_layer_fisher_diagonal

Example:
~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   from torch_secorder.core.per_layer_curvature import per_layer_fisher_diagonal

   # Define a simple model with multiple layers
   class SimpleModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.linear1 = nn.Linear(10, 5)
           self.linear2 = nn.Linear(5, 3)
       def forward(self, x):
           x = F.relu(self.linear1(x))
           return self.linear2(x)

   # Generate synthetic data
   torch.manual_seed(42)
   X = torch.randn(8, 10)
   y = torch.randint(0, 3, (8,))  # 3 classes

   # Instantiate the model
   model = SimpleModel()

   # Compute per-layer Fisher diagonal
   # This extracts block-diagonal approximations of the Fisher Information Matrix
   # These blocks are the basis for K-FAC approximations
   layer_fisher_diagonals = per_layer_fisher_diagonal(model, F.cross_entropy, X, y)

   print("Per-Layer Fisher Diagonal (K-FAC basis):")
   for layer_name, diag in layer_fisher_diagonals.items():
       print(f"Layer: {layer_name}, Shape: {diag.shape}, First few values: {diag.flatten()[:5].tolist()}")

   print("\nK-FAC would further decompose these blocks into Kronecker products of smaller matrices.")

Layer Curvature Statistics
--------------------------

.. autofunction:: get_layer_curvature_stats

Example:
~~~~~~~~

.. code-block:: python

   import torch
   from torch_secorder.core.per_layer_curvature import get_layer_curvature_stats

   # Using the layer_hessians from the previous example
   stats = get_layer_curvature_stats(layer_hessians)

   # Print statistics for each layer
   for layer_name, layer_stats in stats.items():
       print(f"\n{layer_name} statistics:")
       print(f"  Mean: {layer_stats['mean']:.4f}")
       print(f"  Std:  {layer_stats['std']:.4f}")
       print(f"  Max:  {layer_stats['max']:.4f}")
       print(f"  Min:  {layer_stats['min']:.4f}")

Notes
-----

1. The per-layer curvature computations provide a block-diagonal approximation of the full
   Hessian/Fisher matrix, where each block corresponds to a layer's parameters.

2. This approximation is useful for:
   - Understanding the curvature of individual layers
   - Implementing layer-wise optimization strategies
   - Diagnosing training issues at the layer level
   - Reducing computational complexity compared to full matrix computations

3. The `layer_types` parameter allows you to specify which types of layers to include
   in the computation. By default, it includes `nn.Linear` and `nn.Conv2d` layers.

4. The `create_graph` parameter allows for computing higher-order derivatives, which is
   useful in meta-learning or other advanced scenarios.
