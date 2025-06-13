"""Functions for computing the diagonal elements of Fisher Information Matrices.

This module provides implementations for extracting diagonal elements from both
Empirical and Generalized Fisher Information Matrices, which are common approximations
of the Hessian for various machine learning tasks.
"""

from typing import Callable, List

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


def empirical_fisher_diagonal(
    model: Module,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    inputs: Tensor,
    targets: Tensor,
) -> List[Tensor]:
    """Compute the diagonal elements of the Empirical Fisher Information Matrix.

    The Empirical Fisher is approximated by the squared gradients of the loss
    with respect to the model parameters. This function computes the diagonal
    elements of this approximation for each parameter.

    Args:
        model: The PyTorch model.
        loss_fn: The loss function, e.g., ``nn.CrossEntropyLoss()`` or ``nn.MSELoss()``.
        inputs: Input tensor to the model.
        targets: Target tensor for the loss function.

    Returns:
        A list of tensors, each containing the diagonal elements of the EFIM
        for the corresponding parameter.
    """
    # Compute gradients of the loss with respect to parameters
    loss = loss_fn(model(inputs), targets)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

    efim_diagonal = []
    for g in grads:
        if g is not None:
            # The diagonal of EFIM is typically g_i^2
            efim_diagonal.append(g.pow(2))
        else:
            efim_diagonal.append(torch.zeros_like(g))
    return efim_diagonal


def generalized_fisher_diagonal(
    model: Module,
    outputs: Tensor,
    targets: Tensor,
    loss_type: str = "nll",
    create_graph: bool = False,
) -> List[Tensor]:
    """Compute the diagonal elements of the Generalized Fisher Information Matrix.

    The Generalized Fisher Information Matrix (GFIM) is defined as the expectation of the
    outer product of the gradients of the log-likelihood with respect to the parameters.
    This function computes the diagonal elements of this approximation.

    Args:
        model: The PyTorch model.
        outputs: The raw outputs (e.g., logits) from the model.
        targets: The target tensor (e.g., class labels or regression targets).
        loss_type: Specifies the type of likelihood. Currently supports 'nll' (Negative Log Likelihood).
        create_graph: If True, the computational graph will be constructed,
                     allowing for higher-order derivatives.

    Returns:
        A list of tensors, each containing the diagonal elements of the GFIM
        for the corresponding parameter.

    Raises:
        NotImplementedError: If an unsupported `loss_type` or output shape is provided.
    """
    if loss_type.lower() == "nll":
        # For classification, assuming outputs are logits and targets are class indices.
        # The gradient of the negative log-likelihood w.r.t. parameters is used.
        # For GFIM, F = E[grad(log p(y|x,w)) grad(log p(y|x,w))^T].
        # A practical approximation for the diagonal of GFIM (especially for classification)
        # is the squared gradients of the (negative) log-likelihood.

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

            gfim_diagonal = []
            for g in grads:
                if g is not None:
                    # The diagonal of GFIM is typically g_i^2
                    gfim_diagonal.append(g.pow(2))
                else:
                    gfim_diagonal.append(torch.zeros_like(g))
            return gfim_diagonal
        else:
            raise NotImplementedError(
                "Generalized Fisher Diagonal for non-NLL loss types or unsupported output/target shapes is not yet implemented."
            )
    else:
        raise NotImplementedError(
            f"Loss type '{loss_type}' not supported for Generalized Fisher Diagonal."
        )
