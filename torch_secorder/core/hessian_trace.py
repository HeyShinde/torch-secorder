"""Module for estimating the trace of the Hessian matrix.

This module provides functions to estimate the trace of the Hessian matrix
using various methods, including Hutchinson's method (HVP-based) and
summing the diagonal elements (diagonal-based).
"""

from typing import Callable, List

from torch import Tensor
from torch.nn import Module

# Import hessian_diagonal from its new location to use it for trace estimation
from .hessian_diagonal import hessian_diagonal


def hessian_trace(
    func: Callable[[], Tensor],
    params: List[Tensor],
    num_samples: int = 1,
    create_graph: bool = False,
    strict: bool = False,
) -> Tensor:
    """Compute the trace of the Hessian matrix by summing the diagonal elements.

    Args:
        func: A callable that returns a scalar tensor (the loss).
        params: List of parameters with respect to which the Hessian is computed.
        num_samples: Ignored (kept for API compatibility).
        create_graph: If True, the computational graph will be constructed,
                     allowing for higher-order derivatives.
        strict: If True, an error will be raised if any parameter requires grad
                but has no gradient.

    Returns:
        A tensor containing the trace of the Hessian.
    """
    diag = hessian_diagonal(func, params, create_graph=create_graph, strict=strict)
    return sum([d.sum() for d in diag])


def model_hessian_trace(
    model: Module,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    inputs: Tensor,
    targets: Tensor,
    num_samples: int = 1,
    create_graph: bool = False,
    strict: bool = False,
) -> Tensor:
    """Compute the trace of the Hessian for a model's loss function.

    A convenience function to estimate the trace of the Hessian of a model's loss
    with respect to its parameters.

    Args:
        model: The PyTorch model.
        loss_fn: The loss function, e.g., ``nn.MSELoss()`` or ``F.cross_entropy``.
        inputs: Input tensor to the model.
        targets: Target tensor for the loss function.
        num_samples: Number of random vectors for Hutchinson's method (if used).
                     Ignored for diagonal-based trace.
        create_graph: If True, the computational graph will be constructed,
                     allowing for higher-order derivatives.
        strict: If True, an error will be raised if any parameter requires grad
                but has no gradient.

    Returns:
        A tensor containing the estimated trace of the Hessian.
    """

    def loss_func():
        outputs = model(inputs)
        return loss_fn(outputs, targets)

    return hessian_trace(
        loss_func,
        list(model.parameters()),
        create_graph=create_graph,
        strict=strict,
    )
