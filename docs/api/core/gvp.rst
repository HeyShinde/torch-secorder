Jacobian-Vector Product (GVP) Module
=======================================

The GVP module provides implementations of Jacobian-vector products (JVP), vector-Jacobian products (VJP),
and related utilities.

JVP (Jacobian-Vector Product)
-----------------------------

.. autofunction:: torch_secorder.core.gvp.jvp

Computes the Jacobian-vector product for a given function and parameters.

Example:
~~~~~~~~

.. code-block:: python

    import torch
    from torch_secorder.core.gvp import jvp

    def func():
        return torch.stack([x[0] ** 2, 3 * x[1] ** 2])

    x = torch.tensor([1.0, 2.0], requires_grad=True)
    v = torch.tensor([0.5, -1.0])
    jvp_result = jvp(func, [x], v)

VJP (Vector-Jacobian Product)
-----------------------------

.. autofunction:: torch_secorder.core.gvp.vjp

Computes the vector-Jacobian product for a given function and parameters.

Example:
~~~~~~~~

.. code-block:: python

    import torch
    from torch_secorder.core.gvp import vjp

    def func():
        return torch.stack([x[0] ** 2, 3 * x[1] ** 2])

    x = torch.tensor([1.0, 2.0], requires_grad=True)
    v = torch.tensor([0.5, -1.0])
    vjp_result = vjp(func, [x], v)

Model JVP
---------

.. autofunction:: torch_secorder.core.gvp.model_jvp

A convenience wrapper for computing JVP with respect to a model's parameters.

Example:
~~~~~~~~

.. code-block:: python

    import torch
    import torch.nn as nn
    from torch_secorder.core.gvp import model_jvp

    model = nn.Linear(10, 1)
    x = torch.randn(1, 10)
    v = [torch.randn_like(p) for p in model.parameters()]
    jvp_result = model_jvp(model, x, v)

Model VJP
---------

.. autofunction:: torch_secorder.core.gvp.model_vjp

A convenience wrapper for computing VJP with respect to a model's parameters.

Example:
~~~~~~~~

.. code-block:: python

    import torch
    import torch.nn as nn
    from torch_secorder.core.gvp import model_vjp

    model = nn.Linear(10, 1)
    x = torch.randn(1, 10)
    v = torch.randn(1, 1)
    vjp_result = model_vjp(model, x, v)

Batch JVP
---------

.. autofunction:: torch_secorder.core.gvp.batch_jvp

Computes JVPs for a batch of vectors efficiently.

Example:
~~~~~~~~

.. code-block:: python

    import torch
    from torch_secorder.core.gvp import batch_jvp

    def func():
        return torch.stack([x[0] ** 2, 3 * x[1] ** 2])

    x = torch.tensor([1.0, 2.0], requires_grad=True)
    vs = torch.stack([
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0])
    ])
    batch_result = batch_jvp(func, [x], vs)

Batch VJP
---------

.. autofunction:: torch_secorder.core.gvp.batch_vjp

Computes VJPs for a batch of vectors efficiently.

Example:
~~~~~~~~

.. code-block:: python

    import torch
    from torch_secorder.core.gvp import batch_vjp

    def func():
        return torch.stack([x[0] ** 2, 3 * x[1] ** 2])

    x = torch.tensor([1.0, 2.0], requires_grad=True)
    vs = torch.stack([
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0])
    ])
    batch_result = batch_vjp(func, [x], vs)

Full Jacobian
-------------

.. autofunction:: torch_secorder.core.gvp.full_jacobian

Computes the full Jacobian matrix for a given function and parameters.

Example:
~~~~~~~~

.. code-block:: python

    import torch
    from torch_secorder.core.gvp import full_jacobian

    def func():
        return torch.stack([x[0] ** 2, 3 * x[1] ** 2])

    x = torch.tensor([1.0, 2.0], requires_grad=True)
    jac = full_jacobian(func, [x])

Notes
-----

1.  **JVP (`jvp`, `model_jvp`, `batch_jvp`)**: These functions compute the product of the Jacobian matrix with a vector (or batch of vectors). This is generally more efficient than computing the full Jacobian when only the product is needed.

2.  **VJP (`vjp`, `model_vjp`, `batch_vjp`)**: These functions compute the product of a vector (or batch of vectors) with the transpose of the Jacobian matrix. This is also known as a reverse-mode differentiation and is the basis for backpropagation.

3.  **`create_graph` Parameter**: When `create_graph=True` is set, a computational graph of the derivative itself is constructed. This allows for computing higher-order derivatives (e.g., Hessian-vector products from JVPs/VJPs).

4.  **`allow_unused=True`**: This parameter in `torch.autograd.grad` is used to allow gradients to be computed for parameters that might not be part of the computational graph for a specific output. If a parameter does not affect the output, its gradient will be `None`, and the functions handle this by replacing `None` with zero tensors.

5.  **Batch Computations**: The `batch_jvp` and `batch_vjp` functions provide an efficient way to compute JVPs and VJPs for multiple vectors in a single call, which can be beneficial for performance compared to looping through individual vector computations.

6.  **Full Jacobian (`full_jacobian`)**: While JVP and VJP are efficient for products, `full_jacobian` computes the entire Jacobian matrix. This can be memory-intensive for models with many inputs/outputs or parameters but is useful when the entire matrix is required for analysis.
