"""Jacobian-Vector Product (JVP) and Vector-Jacobian Product (VJP) utilities.

This module provides functions to compute JVPs and VJPs efficiently using PyTorch's autograd system.
Both functional and model-based APIs are provided.
"""

from typing import Callable, List, Union

import torch
import torch.nn as nn


def jvp(
    func: Callable[[], torch.Tensor],
    params: List[torch.Tensor],
    v: Union[torch.Tensor, List[torch.Tensor]],
    create_graph: bool = False,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Compute the Jacobian-vector product (JVP): J v.

    Args:
        func: A callable that returns a tensor output (can be vector-valued).
        params: List of parameters with respect to which to compute the Jacobian.
        v: Vector to multiply with the Jacobian. Can be a single tensor or a list of tensors
           matching the structure of params.
        create_graph: If True, graph of the derivative will be constructed, allowing to compute higher order derivative products.

    Returns:
        The JVP (same shape as the output of func).
    """
    if isinstance(v, torch.Tensor):
        v = [v]
    output = func()
    flat_output = output.reshape(-1)
    jvp_result = torch.zeros_like(flat_output)
    for i in range(flat_output.shape[0]):
        grad = torch.autograd.grad(
            flat_output[i],
            params,
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=True,
        )
        grad = [
            (g if g is not None else torch.zeros_like(p)) for g, p in zip(grad, params)
        ]
        jvp_result[i] = sum([(g * v_).sum() for g, v_ in zip(grad, v)])
    return jvp_result.reshape(output.shape)


def vjp(
    func: Callable[[], torch.Tensor],
    params: List[torch.Tensor],
    v: torch.Tensor,
    create_graph: bool = False,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Compute the vector-Jacobian product (VJP): v^T J.

    Args:
        func: A callable that returns a tensor output (can be vector-valued).
        params: List of parameters with respect to which to compute the Jacobian.
        v: Vector to multiply with the Jacobian (should match the output shape of func).
        create_graph: If True, graph of the derivative will be constructed, allowing to compute higher order derivative products.

    Returns:
        The VJP (list of tensors matching the structure of params).
    """
    output = func()
    v = v.reshape_as(output)
    grads = torch.autograd.grad(
        output, params, grad_outputs=v, create_graph=create_graph, allow_unused=True
    )
    grads = [
        (g if g is not None else torch.zeros_like(p)) for g, p in zip(grads, params)
    ]
    return grads[0] if len(grads) == 1 else grads


def model_jvp(
    model: nn.Module,
    x: torch.Tensor,
    v: Union[torch.Tensor, List[torch.Tensor]],
    create_graph: bool = False,
) -> torch.Tensor:
    """Compute the JVP for a model's output with respect to its parameters.

    Args:
        model: The PyTorch model.
        x: Input tensor.
        v: Vector to multiply with the Jacobian (should match the structure of model.parameters()).
        create_graph: If True, graph of the derivative will be constructed.

    Returns:
        The JVP (same shape as the model output).
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    params = list(model.parameters())

    def forward():
        return model(x)

    return jvp(forward, params, v, create_graph=create_graph)


def model_vjp(
    model: nn.Module,
    x: torch.Tensor,
    v: torch.Tensor,
    create_graph: bool = False,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Compute the VJP for a model's output with respect to its parameters.

    Args:
        model: The PyTorch model.
        x: Input tensor.
        v: Vector to multiply with the Jacobian (should match the output shape of model(x)).
        create_graph: If True, graph of the derivative will be constructed.

    Returns:
        The VJP (list of tensors matching the structure of model.parameters()).
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    params = list(model.parameters())

    def forward():
        return model(x)

    return vjp(forward, params, v, create_graph=create_graph)


def batch_jvp(
    func: Callable[[], torch.Tensor],
    params: List[torch.Tensor],
    vs: Union[torch.Tensor, List[torch.Tensor]],
    create_graph: bool = False,
) -> torch.Tensor:
    """Compute a batch of Jacobian-vector products (JVPs).

    Args:
        func: A callable that returns a tensor output (can be vector-valued).
        params: List of parameters with respect to which to compute the Jacobian.
        vs: Batch of vectors to multiply with the Jacobian. Should be a tensor of shape (batch, ...) or a list of such tensors.
        create_graph: If True, graph of the derivative will be constructed.

    Returns:
        Tensor of shape (batch, ...) with the JVPs for each vector in the batch.
    """
    if isinstance(vs, torch.Tensor):
        vs = [vs]
    batch_size = vs[0].shape[0]
    results = []
    for i in range(batch_size):
        v_i = [v[i] for v in vs]
        results.append(jvp(func, params, v_i, create_graph=create_graph))
    return torch.stack(results)


def batch_vjp(
    func: Callable[[], torch.Tensor],
    params: List[torch.Tensor],
    vs: torch.Tensor,
    create_graph: bool = False,
) -> List[torch.Tensor]:
    """Compute a batch of vector-Jacobian products (VJPs).

    Args:
        func: A callable that returns a tensor output (can be vector-valued).
        params: List of parameters with respect to which to compute the Jacobian.
        vs: Batch of vectors to multiply with the Jacobian (should match the output shape of func, with batch dimension first).
        create_graph: If True, graph of the derivative will be constructed.

    Returns:
        List of tensors, each of shape (batch, ...) matching the structure of params.
    """
    batch_size = vs.shape[0]
    results: list[list[torch.Tensor]] = [[] for _ in params]
    for i in range(batch_size):
        v_i = vs[i]
        vjp_i = vjp(func, params, v_i, create_graph=create_graph)
        if isinstance(vjp_i, torch.Tensor):
            vjp_i = [vjp_i]
        for j, vj in enumerate(vjp_i):
            results[j].append(vj)
    return [torch.stack(r) for r in results]


def full_jacobian(
    func: Callable[[], torch.Tensor],
    params: List[torch.Tensor],
    create_graph: bool = False,
) -> List[torch.Tensor]:
    """Compute the full Jacobian matrix of func with respect to params.

    Args:
        func: A callable that returns a tensor output (can be vector-valued).
        params: List of parameters with respect to which to compute the Jacobian.
        create_graph: If True, graph of the derivative will be constructed.

    Returns:
        List of Jacobian tensors, one for each parameter, with shape (output_dim, param_shape).
    """
    output = func()
    flat_output = output.reshape(-1)
    jacobians = []
    for p in params:
        jac_rows = []
        for i in range(flat_output.shape[0]):
            grad = torch.autograd.grad(
                flat_output[i],
                p,
                retain_graph=True,
                create_graph=create_graph,
                allow_unused=True,
            )[0]
            if grad is None:
                grad = torch.zeros_like(p)
            jac_row = grad.reshape(-1)
            jac_rows.append(jac_row)
        jac = torch.stack(jac_rows, dim=0)
        jacobians.append(jac)
    return jacobians
