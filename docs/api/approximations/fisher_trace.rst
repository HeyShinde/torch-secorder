.. _fisher_trace_api:

Fisher Trace Module
===================

.. currentmodule:: torch_secorder.approximations.fisher_trace

The Fisher Trace module provides functions to estimate the trace of Fisher Information Matrices.

Empirical Fisher Trace
----------------------

.. autofunction:: empirical_fisher_trace

Example:
~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   from torch_secorder.approximations.fisher_trace import empirical_fisher_trace

   # Define a simple model and loss function
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

   # Estimate the Empirical Fisher Trace
   efim_trace = empirical_fisher_trace(model, loss_fn, inputs, targets)

   print("Empirical Fisher Trace:", efim_trace)

Generalized Fisher Trace
------------------------

.. autofunction:: generalized_fisher_trace

Example:
~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   from torch_secorder.approximations.fisher_trace import generalized_fisher_trace

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

   # Estimate the Generalized Fisher Trace (NLL loss type)
   gfim_trace = generalized_fisher_trace(model, outputs, targets, loss_type="nll")

   print("Generalized Fisher Trace:", gfim_trace)

Notes
-----

1. The Fisher trace computation can be used to approximate the Hessian's trace for second-order optimization.

2. The `create_graph` parameter for `generalized_fisher_trace` allows for computing higher-order derivatives if needed, but this increases memory usage.

3. The `num_samples` parameter for `empirical_fisher_trace` is effectively ignored in the current implementation when the trace is computed directly by summing squared gradients, which is exact for EFIM.
