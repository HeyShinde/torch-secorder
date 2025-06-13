Hessian Trace Module
====================

.. currentmodule:: torch_secorder.core.hessian_trace

The Hessian Trace module provides functions to estimate the trace of the Hessian matrix.

Hessian Trace
-------------

.. autofunction:: hessian_trace

Estimates the trace of the Hessian matrix by summing the diagonal elements, or using Hutchinson's method based on the `num_samples` parameter.

Example:
~~~~~~~~

.. code-block:: python

    import torch
    from torch_secorder.core.hessian_trace import hessian_trace

    x = torch.tensor([1.0, 2.0], requires_grad=True)

    def quadratic():
        return x.pow(2).sum()

    # Estimate the trace with 1000 samples
    trace = hessian_trace(quadratic, [x], num_samples=1000)
    print(trace)  # Should be close to 4.0 (trace of 2I)

Model Hessian Trace
-------------------

.. autofunction:: model_hessian_trace

A convenience function to estimate the trace of the Hessian of a model's loss with respect to its parameters.

Example:
~~~~~~~~

.. code-block:: python

    import torch
    import torch.nn as nn
    from torch_secorder.core.hessian_trace import model_hessian_trace

    # Create a simple neural network
    model = nn.Linear(10, 1)
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)

    # Estimate the trace of the Hessian for the MSE loss
    trace = model_hessian_trace(model, nn.functional.mse_loss, x, y)
    print(f"Estimated Hessian Trace: {trace}")

Trace Estimation Methods
------------------------

The library provides two methods for estimating the trace of the Hessian matrix:

1. **HVP-based Trace** (from :mod:`torch_secorder.core.hvp`):
   - Uses Hutchinson's method with Hessian-vector products
   - More memory-efficient for large models
   - Better suited when computing the full diagonal is expensive
   - Uses random vectors to estimate the trace

2. **Diagonal-based Trace** (from this module):
   - Computes the exact diagonal elements of the Hessian (via `hessian_diagonal`)
   - More accurate but more computationally expensive
   - Better suited for smaller models
   - Can use custom vectors for more control

The choice between these methods depends on your specific needs:
- Use HVP-based trace for large models where memory efficiency is crucial
- Use diagonal-based trace when accuracy is more important than computational cost
- Both methods are related (they compute the same quantity) but use different approaches
- The HVP version is more memory-efficient but may be less accurate
- The diagonal version is more accurate but requires more computation

Notes
-----

1. The Hessian trace computation using this module by default sums the diagonal elements obtained from `hessian_diagonal`. For a Hutchinson-style trace estimation, use the `hessian_trace` function from the `hvp` module.

2. The trace estimation can be made more accurate by increasing the number of samples for Hutchinson's method, but this comes at the cost of increased computation time.

3. The `create_graph` parameter allows for computing higher-order derivatives if needed, but this increases memory usage.

4. The `strict` parameter controls whether an error should be raised if any parameter requires gradients but has none.
