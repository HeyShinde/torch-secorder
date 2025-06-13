"""Utilities for extracting per-layer curvature information from neural networks.

This module provides functions to compute block-diagonal approximations of the Hessian
and Fisher Information matrices, where each block corresponds to a layer's parameters.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from torch_secorder.core.hessian_diagonal import hessian_diagonal
from torch_secorder.core.utils import get_params_by_module_type


def per_layer_hessian_diagonal(
    model: nn.Module,
    loss_fn: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    layer_types: Optional[List[type]] = None,
    create_graph: bool = False,
) -> Dict[str, torch.Tensor]:
    """Compute the diagonal of the Hessian matrix for each layer separately.

    This function computes the diagonal elements of the Hessian matrix for each layer
    in the model independently, treating other layers' parameters as fixed. This provides
    a block-diagonal approximation of the full Hessian.

    Args:
        model: The neural network model.
        loss_fn: The loss function used for training.
        inputs: Input tensor for the model.
        targets: Target tensor for the loss function.
        layer_types: List of layer types to include. If None, includes all layers.
        create_graph: Whether to create the computational graph for higher-order derivatives.

    Returns:
        Dictionary mapping layer names to their Hessian diagonal tensors.
    """
    if layer_types is None:
        layer_types = [nn.Linear, nn.Conv2d]  # Default to common layer types

    layer_params = get_params_by_module_type(model, layer_types)
    layer_hessians = {}

    for layer_name, params in layer_params.items():
        # Compute loss with only this layer's parameters requiring gradients
        original_requires_grad = {}
        for n, p in model.named_parameters():
            original_requires_grad[n] = p.requires_grad
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)

        # Compute Hessian diagonal for this layer
        hessian_diag = hessian_diagonal(
            lambda: loss_fn(model(inputs), targets), params, create_graph=create_graph
        )
        # Flatten the list of diagonal tensors for the layer into a single tensor
        layer_hessians[layer_name] = torch.cat([d.flatten() for d in hessian_diag])

        # Restore original requires_grad states
        for n, p in model.named_parameters():
            p.requires_grad_(original_requires_grad[n])

    return layer_hessians


def per_layer_fisher_diagonal(
    model: nn.Module,
    loss_fn: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    layer_types: Optional[List[type]] = None,
    create_graph: bool = False,
) -> Dict[str, torch.Tensor]:
    """Compute the diagonal of the Fisher Information matrix for each layer separately.

    This function computes the diagonal elements of the Fisher Information matrix for each
    layer in the model independently, treating other layers' parameters as fixed. This
    provides a block-diagonal approximation of the full Fisher matrix.

    Args:
        model: The neural network model.
        loss_fn: The loss function used for training.
        inputs: Input tensor for the model.
        targets: Target tensor for the loss function.
        layer_types: List of layer types to include. If None, includes all layers.
        create_graph: Whether to create the computational graph for higher-order derivatives.

    Returns:
        Dictionary mapping layer names to their Fisher diagonal tensors.
    """
    if layer_types is None:
        layer_types = [nn.Linear, nn.Conv2d]  # Default to common layer types

    layer_params = get_params_by_module_type(model, layer_types)
    layer_fishers = {}

    for layer_name, params in layer_params.items():
        # Compute loss with only this layer's parameters requiring gradients
        original_requires_grad = {}
        for n, p in model.named_parameters():
            original_requires_grad[n] = p.requires_grad
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)

        # Compute loss and gradients for the current layer
        loss = loss_fn(model(inputs), targets)
        grads = torch.autograd.grad(
            loss, params, create_graph=create_graph, allow_unused=True
        )

        # Compute the diagonal of the Empirical Fisher (g_i^2)
        fisher_diag_list = []
        for g in grads:
            if g is not None:
                fisher_diag_list.append(g.pow(2))
            else:
                # If a parameter has no gradient, append a zero tensor of its shape
                fisher_diag_list.append(torch.zeros_like(params[grads.index(g)]))

        layer_fishers[layer_name] = torch.cat([d.flatten() for d in fisher_diag_list])

        # Restore original requires_grad states
        for n, p in model.named_parameters():
            p.requires_grad_(original_requires_grad[n])

    return layer_fishers


def get_layer_curvature_stats(
    layer_curvatures: Dict[str, torch.Tensor],
) -> Dict[str, Dict[str, float]]:
    """Compute basic statistics for each layer's curvature information.

    Args:
        layer_curvatures: Dictionary mapping layer names to their curvature tensors
                         (Hessian or Fisher diagonals).

    Returns:
        Dictionary mapping layer names to their curvature statistics (mean, std, max, min).
    """
    stats = {}
    for layer_name, curvature in layer_curvatures.items():
        stats[layer_name] = {
            "mean": float(curvature.mean().item()),
            "std": float(curvature.std().item()),
            "max": float(curvature.max().item()),
            "min": float(curvature.min().item()),
        }
    return stats
