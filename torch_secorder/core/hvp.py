"""Hessian-Vector Product (HVP) computation utilities.

This module provides functions to compute Hessian-vector products efficiently
using PyTorch's autograd system. Both exact and approximate methods are provided.
"""

from typing import Callable, List, Union

import torch
import torch.nn as nn


def exact_hvp(
    func: Callable[[], torch.Tensor],
    params: List[torch.Tensor],
    v: Union[torch.Tensor, List[torch.Tensor]],
    create_graph: bool = False,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Compute the exact Hessian-vector product Hv using double backpropagation.

    Args:
        func: A callable that returns a scalar tensor (loss).
        params: List of parameters with respect to which to compute the Hessian.
        v: Vector to multiply with the Hessian. Can be a single tensor or a list of tensors
           matching the structure of params.
        create_graph: If True, the graph used to compute the grad will be constructed,
                     allowing to compute higher order derivative products.

    Returns:
        The Hessian-vector product Hv. Returns a single tensor if v is a single tensor,
        otherwise returns a list of tensors matching the structure of params.
    """
    if isinstance(v, torch.Tensor):
        v = [v]

    # First compute the gradient
    grad = torch.autograd.grad(func(), params, create_graph=True)

    # Then compute the Hessian-vector product
    # Ensure the output is a scalar by summing all elements
    grad_dot_v = sum((g * v_).sum() for g, v_ in zip(grad, v))
    hvp = torch.autograd.grad(
        grad_dot_v,
        params,
        create_graph=create_graph,
    )

    return hvp[0] if len(hvp) == 1 else hvp


def approximate_hvp(
    func: Callable[[], torch.Tensor],
    params: List[torch.Tensor],
    v: Union[torch.Tensor, List[torch.Tensor]],
    num_samples: int = 1,
    damping: float = 0.0,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Compute an approximate Hessian-vector product using finite differences.

    This method uses a finite difference approximation of the Hessian-vector product,
    which can be more memory efficient than the exact computation.

    Args:
        func: A callable that returns a scalar tensor (loss).
        params: List of parameters with respect to which to compute the Hessian.
        v: Vector to multiply with the Hessian. Can be a single tensor or a list of tensors
           matching the structure of params.
        num_samples: Number of samples to use for the approximation.
        damping: Damping term to add to the diagonal of the Hessian (lambda * I).

    Returns:
        The approximate Hessian-vector product Hv. Returns a single tensor if v is a single tensor,
        otherwise returns a list of tensors matching the structure of params.
    """
    if isinstance(v, torch.Tensor):
        v = [v]

    eps = 1e-6  # Small constant for finite differences

    # Compute the gradient at the current point
    grad = torch.autograd.grad(func(), params, create_graph=False)

    # Initialize the result
    hvp = [torch.zeros_like(p) for p in params]

    for _ in range(num_samples):
        # Perturb parameters safely
        with torch.no_grad():
            for p, vec in zip(params, v):
                p.add_(eps * vec)
        # Compute gradient at perturbed point
        grad_perturbed = torch.autograd.grad(func(), params, create_graph=False)
        with torch.no_grad():
            for p, vec in zip(params, v):
                p.sub_(eps * vec)
        # Update HVP approximation
        for h, g, g_p in zip(hvp, grad, grad_perturbed):
            h.add_((g_p - g) / eps)

    # Average over samples
    for h in hvp:
        h.div_(num_samples)

    # Add damping if specified
    if damping > 0:
        for h, vec in zip(hvp, v):
            h.add_(damping * vec)

    return hvp[0] if len(hvp) == 1 else hvp


def model_hvp(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    v: Union[torch.Tensor, List[torch.Tensor]],
    create_graph: bool = False,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Compute the Hessian-vector product for a model's loss function.

    This is a convenience wrapper around exact_hvp that handles the model's
    forward pass and loss computation.

    Args:
        model: The PyTorch model.
        loss_fn: The loss function that takes model output and target as arguments.
        x: Input tensor.
        y: Target tensor.
        v: Vector to multiply with the Hessian.
        create_graph: If True, the graph used to compute the grad will be constructed.

    Returns:
        The Hessian-vector product Hv.
    """
    params = list(model.parameters())

    def loss_func():
        output = model(x)
        return loss_fn(output, y)

    return exact_hvp(loss_func, params, v, create_graph=create_graph)


def gauss_newton_product(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    v: Union[torch.Tensor, List[torch.Tensor]],
    create_graph: bool = False,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Compute the Gauss-Newton matrix-vector product for a model's loss.

    Args:
        model: The PyTorch model.
        loss_fn: The loss function (should be MSE or cross-entropy for GN to be valid).
        x: Input tensor.
        y: Target tensor.
        v: Vector to multiply with the Gauss-Newton matrix.
        create_graph: If True, graph of the derivative will be constructed.

    Returns:
        The Gauss-Newton matrix-vector product (same structure as v).
    """
    params = list(model.parameters())

    def output():
        return model(x)

    out = output()
    out_flat = out.reshape(-1)
    # Compute Jv (J: Jacobian of model output wrt params)
    if isinstance(v, torch.Tensor):
        v = [v]
    jvp_vec = torch.zeros_like(out_flat)
    for i in range(out_flat.shape[0]):
        grad = torch.autograd.grad(
            out_flat[i], params, retain_graph=True, create_graph=True, allow_unused=True
        )
        grad = [
            (g if g is not None else torch.zeros_like(p)) for g, p in zip(grad, params)
        ]
        jvp_vec[i] = sum([(g * v_).sum() for g, v_ in zip(grad, v)])
    # Compute Hessian of loss wrt model output (usually diagonal for MSE, cross-entropy)
    out = out.detach().requires_grad_(True)
    loss = loss_fn(out, y)
    grad_out = torch.autograd.grad(loss, out, create_graph=True)[0]
    # Reshape jvp_vec to match out's shape
    jvp_vec_reshaped = jvp_vec.reshape_as(out)
    gn_prod = torch.autograd.grad(
        grad_out, out, grad_outputs=jvp_vec_reshaped, create_graph=create_graph
    )[0]
    # Backpropagate to parameters
    gn_param = torch.autograd.grad(
        out, params, grad_outputs=gn_prod, create_graph=create_graph, allow_unused=True
    )
    gn_param = [
        (g if g is not None else torch.zeros_like(p)) for g, p in zip(gn_param, params)
    ]
    return gn_param[0] if len(gn_param) == 1 else gn_param


def hessian_trace(
    func: Callable[[], torch.Tensor],
    params: List[torch.Tensor],
    num_samples: int = 10,
    create_graph: bool = False,
    sparse: bool = False,
) -> float:
    """Estimate the trace of the Hessian using Hutchinson's method (random projections).

    Args:
        func: A callable that returns a scalar tensor (loss).
        params: List of parameters with respect to which to compute the Hessian.
        num_samples: Number of random projections to use.
        create_graph: If True, graph of the derivative will be constructed.
        sparse: If True, use sparse random vectors for projections.

    Returns:
        Estimated trace of the Hessian.
    """
    trace = 0.0
    for _ in range(num_samples):
        vs = []
        for p in params:
            if sparse:
                # Sparse Rademacher vector
                v = torch.randint_like(p, low=0, high=2, dtype=torch.float32) * 2 - 1
                mask = torch.rand_like(p) > 0.9  # 10% nonzero
                v = v * mask
            else:
                v = torch.randint_like(p, low=0, high=2, dtype=torch.float32) * 2 - 1
            vs.append(v)
        hv = exact_hvp(func, params, vs, create_graph=create_graph)
        if isinstance(hv, torch.Tensor):
            hv = [hv]
        trace += sum([(h * v).sum().item() for h, v in zip(hv, vs)])
    return trace / num_samples
