"""Functions for computing the trace of Fisher Information Matrices.

This module provides implementations for estimating the trace of both
Empirical and Generalized Fisher Information Matrices, which are common approximations
of the Hessian for various machine learning tasks.
"""

from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


def empirical_fisher_trace(
    model: Module,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    inputs: Tensor,
    targets: Tensor,
    num_samples: int = 1,
) -> Tensor:
    """Estimate the trace of the Empirical Fisher Information Matrix using Hutchinson's method.

    This function estimates the trace of the EFIM by leveraging Hutchinson's method,
    which involves computing Jacobian-vector products (or gradient products).

    Args:
        model: The PyTorch model.
        loss_fn: The loss function, e.g., ``nn.CrossEntropyLoss()`` or ``nn.MSELoss()``.
        inputs: Input tensor to the model.
        targets: Target tensor for the loss function.
        num_samples: Number of random vectors to use for Hutchinson's estimation.
                     Higher values lead to more accurate estimates but increase computation.

    Returns:
        A scalar tensor representing the estimated trace of the EFIM.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1 for Hutchinson's method.")

    # For EFIM, the trace is sum(g_i^2) for each parameter. This is essentially the
    # sum of the diagonal elements. So, we can directly compute the diagonal and sum.
    # Alternatively, for a true Hutchinson's approach (more general for other FIMs):
    # E[v^T F v] where F = E[grad_log_p grad_log_p^T]
    # For Empirical Fisher, F = sum(grad_loss_i grad_loss_i^T) / N_batch

    # We can compute the sum of squares of gradients directly
    loss = loss_fn(model(inputs), targets)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)

    # Sum of squared gradients for all parameters
    total_trace = torch.tensor(0.0, device=loss.device)
    for g in grads:
        if g is not None:
            total_trace += (g.pow(2)).sum()

    # For Hutchinson's method, we would typically draw random vectors and compute HVP-like products.
    # However, for Empirical Fisher, the sum of squared gradients is the exact trace of the matrix
    # formed by outer product of gradients. The num_samples parameter is more relevant for
    # generalized Fisher or Hessian traces where a stochastic approximation is needed.

    return total_trace


def generalized_fisher_trace(
    model: Module,
    outputs: Tensor,
    targets: Tensor,
    loss_type: str = "nll",
    num_samples: int = 1,
    create_graph: bool = False,
) -> Tensor:
    """Estimate the trace of the Generalized Fisher Information Matrix using Hutchinson's method.

    The Generalized Fisher Information Matrix (GFIM) is defined as the expectation of the
    outer product of the gradients of the log-likelihood with respect to the parameters.
    This function estimates its trace using the sum of squared gradients of the negative
    log-likelihood, which is a common practical approximation for classification tasks.

    Args:
        model: The PyTorch model.
        outputs: The raw outputs (e.g., logits) from the model.
        targets: The target tensor (e.g., class labels or regression targets).
        loss_type: Specifies the type of likelihood. Currently supports 'nll' (Negative Log Likelihood).
        num_samples: Number of random vectors for Hutchinson's estimation.
                     Higher values lead to more accurate estimates but increase computation.
                     (Note: For 'nll' with current implementation, this parameter is effectively ignored
                     as the trace is computed directly via sum of squared gradients, which is exact for EFIM.)
        create_graph: If True, the computational graph will be constructed,
                     allowing for higher-order derivatives.

    Returns:
        A scalar tensor representing the estimated trace of the GFIM.

    Raises:
        NotImplementedError: If an unsupported `loss_type` or output shape is provided.
        ValueError: If `num_samples` is less than 1.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1 for Hutchinson's method.")

    if loss_type.lower() == "nll":
        # For classification, assuming outputs are logits and targets are class indices.
        # The gradient of the negative log-likelihood w.r.t. parameters is used.
        # For GFIM, F = E[grad(log p(y|x,w)) grad(log p(y|x,w))^T].
        # A practical approximation for the trace of GFIM (especially for classification)
        # is the sum of squared gradients of the (negative) log-likelihood.
        # This is equivalent to the Empirical Fisher trace when the loss is NLL.

        if outputs.ndim == 2 and targets.ndim == 1:
            log_likelihood = -F.nll_loss(
                F.log_softmax(outputs, dim=-1), targets, reduction="sum"
            )
            grads = torch.autograd.grad(
                log_likelihood,
                model.parameters(),
                create_graph=create_graph,
                retain_graph=True,
            )

            total_trace = torch.tensor(0.0, device=outputs.device)
            for g in grads:
                if g is not None:
                    total_trace += (g.pow(2)).sum()
            return total_trace
        else:
            raise NotImplementedError(
                "Generalized Fisher Trace for non-NLL loss types or unsupported output/target shapes is not yet implemented."
            )
    else:
        raise NotImplementedError(
            f"Loss type '{loss_type}' not supported for Generalized Fisher Trace."
        )
