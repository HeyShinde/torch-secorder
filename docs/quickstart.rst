Quickstart Guide
================

This guide will help you get started with torch-secorder quickly.

Basic Usage
-----------

Computing Hessian-Vector Products
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    import torch.nn as nn
    from torch_secorder.core import hvp

    # Create a simple model
    model = nn.Linear(10, 1)
    x = torch.randn(1, 10)
    y = torch.randn(1, 1)

    # Compute exact HVP
    v = [torch.randn_like(p) for p in model.parameters()]
    hvp_result = hvp.model_hvp(
        model,
        nn.MSELoss(),
        x,
        y,
        v
    )

    # Or use approximate HVP for memory efficiency
    approx_hvp = hvp.approximate_hvp(
        lambda: nn.MSELoss()(model(x), y),
        list(model.parameters()),
        v,
        num_samples=10
    )

Computing Jacobian-Vector Products
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from torch_secorder.core import gvp

    # Compute JVP
    jvp_result = gvp.model_jvp(model, x, v)

    # Compute VJP
    vjp_result = gvp.model_vjp(model, x, torch.randn(1, 1))

    # Compute batch JVP
    vs = torch.stack([torch.randn_like(p) for p in model.parameters()])
    batch_jvp_result = gvp.batch_jvp(
        lambda: model(x),
        list(model.parameters()),
        vs
    )

Advanced Usage
--------------

Gauss-Newton Products
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Compute Gauss-Newton matrix-vector product
    gn_result = hvp.gauss_newton_product(
        model,
        nn.MSELoss(),
        x,
        y,
        v
    )

Hessian Trace Estimation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Estimate Hessian trace
    trace = hvp.hessian_trace(
        lambda: nn.MSELoss()(model(x), y),
        list(model.parameters()),
        num_samples=10,
        sparse=True
    )

Full Jacobian Computation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Compute full Jacobian
    jac = gvp.full_jacobian(
        lambda: model(x),
        list(model.parameters())
    )

Best Practices
--------------

1. Use approximate HVP for large models to save memory
2. Use batch operations when computing multiple JVPs/VJPs
3. Use sparse projections for Hessian trace estimation with large models
4. Consider using Gauss-Newton products instead of full Hessian when appropriate
