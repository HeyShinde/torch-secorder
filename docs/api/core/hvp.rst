Hessian-Vector Product (HVP) Module
====================================

The HVP module provides efficient implementations of Hessian-vector products and related utilities.

Exact HVP
---------

.. autofunction:: torch_secorder.core.hvp.exact_hvp

The exact HVP computation uses double backpropagation to compute the Hessian-vector product.
This method is accurate but can be memory-intensive for large models.

Example:
~~~~~~~~

.. code-block:: python

    import torch
    from torch_secorder.core.hvp import exact_hvp

    def loss_func():
        return model(x).sum()

    v = [torch.randn_like(p) for p in model.parameters()]
    hvp_result = exact_hvp(loss_func, list(model.parameters()), v)

Approximate HVP
---------------

.. autofunction:: torch_secorder.core.hvp.approximate_hvp

The approximate HVP uses finite differences to estimate the Hessian-vector product.
This method is more memory-efficient but less accurate than the exact computation.

Example:
~~~~~~~~

.. code-block:: python

    import torch
    from torch_secorder.core.hvp import approximate_hvp

    def loss_func():
        return model(x).sum()

    v = [torch.randn_like(p) for p in model.parameters()]
    hvp_result = approximate_hvp(
        loss_func,
        list(model.parameters()),
        v,
        num_samples=10,
        damping=0.1
    )

Model HVP
---------

.. autofunction:: torch_secorder.core.hvp.model_hvp

A convenience wrapper for computing HVP with respect to a model's loss function.

Example:
~~~~~~~~

.. code-block:: python

    import torch
    import torch.nn as nn
    from torch_secorder.core.hvp import model_hvp

    model = nn.Linear(10, 1)
    x = torch.randn(1, 10)
    y = torch.randn(1, 1)
    v = [torch.randn_like(p) for p in model.parameters()]

    hvp_result = model_hvp(
        model,
        nn.MSELoss(),
        x,
        y,
        v
    )

Gauss-Newton Product
--------------------

.. autofunction:: torch_secorder.core.hvp.gauss_newton_product

Computes the Gauss-Newton matrix-vector product, which is a positive semi-definite approximation
of the Hessian matrix.

Example:
~~~~~~~~

.. code-block:: python

    import torch
    import torch.nn as nn
    from torch_secorder.core.hvp import gauss_newton_product

    model = nn.Linear(10, 1)
    x = torch.randn(1, 10)
    y = torch.randn(1, 1)
    v = [torch.randn_like(p) for p in model.parameters()]

    gn_result = gauss_newton_product(
        model,
        nn.MSELoss(),
        x,
        y,
        v
    )

Trace Estimation Methods
------------------------

The library provides two methods for estimating the trace of the Hessian matrix:

1. **HVP-based Trace** (from this module):
   - Uses Hutchinson's method with Hessian-vector products
   - More memory-efficient for large models
   - Better suited when computing the full diagonal is expensive
   - Uses random vectors to estimate the trace

2. **Diagonal-based Trace** (from :mod:`torch_secorder.core.hessian_trace`):
   - Computes the exact diagonal elements of the Hessian
   - More accurate but more computationally expensive
   - Better suited for smaller models
   - Can use custom vectors for more control

The choice between these methods depends on your specific needs:
- Use HVP-based trace for large models where memory efficiency is crucial
- Use diagonal-based trace when accuracy is more important than computational cost
- Both methods are related (they compute the same quantity) but use different approaches
- The HVP version is more memory-efficient but may be less accurate
- The diagonal version is more accurate but requires more computation

Hessian Trace
-------------

.. autofunction:: torch_secorder.core.hvp.hessian_trace

Estimates the trace of the Hessian matrix using Hutchinson's method with random projections.

Example:
~~~~~~~~

.. code-block:: python

    import torch
    from torch_secorder.core.hvp import hessian_trace

    def loss_func():
        return model(x).sum()

    trace = hessian_trace(
        loss_func,
        list(model.parameters()),
        num_samples=10,
        sparse=True
    )
