.. _fisher_diagonal_api:

Fisher Diagonal Module
======================

.. currentmodule:: torch_secorder.approximations.fisher_diagonal

The Fisher Diagonal module provides functions to compute the diagonal elements of Fisher Information Matrices.

Empirical Fisher Diagonal
-------------------------

.. autofunction:: empirical_fisher_diagonal

Example:
~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   from torch_secorder.approximations.fisher_diagonal import empirical_fisher_diagonal

   # Define a simple model and loss function
   class SimpleModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.linear = nn.Linear(10, 2)

       def forward(self, x):
           return self.linear(x)

   model = SimpleModel()
   loss_fn = nn.CrossEntropyLoss()
   inputs = torch.randn(5, 10)  # Batch size 5, input dim 10
   targets = torch.randint(0, 2, (5,)) # Batch size 5, 2 classes

   # Compute the Empirical Fisher Diagonal
   efim_diagonal = empirical_fisher_diagonal(model, loss_fn, inputs, targets)

   print("Empirical Fisher Diagonal:", efim_diagonal)

Generalized Fisher Diagonal
---------------------------

.. autofunction:: generalized_fisher_diagonal

Example:
~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   from torch_secorder.approximations.fisher_diagonal import generalized_fisher_diagonal

   # Define a simple model
   class SimpleModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.linear = nn.Linear(10, 2)

       def forward(self, x):
           return self.linear(x)

   model = SimpleModel()
   inputs = torch.randn(5, 10)
   outputs = model(inputs)  # Logits
   targets = torch.randint(0, 2, (5,)) # Class labels

   # Compute the Generalized Fisher Diagonal (NLL loss type)
   gfim_diagonal = generalized_fisher_diagonal(model, outputs, targets, loss_type="nll")

   print("Generalized Fisher Diagonal:", gfim_diagonal)

Notes
-----

1. The Fisher diagonal computation can be used to approximate the Hessian's diagonal for second-order optimization.

2. The `create_graph` parameter for `generalized_fisher_diagonal` allows for computing higher-order derivatives if needed, but this increases memory usage.
