"""Module for visualizing loss landscapes.

This module provides utilities to plot 1D slices and 2D contours of the loss landscape
along specified directions, which can help in understanding optimization dynamics and
model generalization.
"""

from typing import Callable, List, Tuple

import torch
import torch.nn as nn


def compute_loss_surface_1d(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    direction: List[torch.Tensor],
    alpha_range: Tuple[float, float] = (-1.0, 1.0),
    num_points: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes loss values along a 1D slice in the parameter space.

    Args:
        model: The PyTorch model.
        loss_fn: The loss function.
        inputs: Input tensor to the model.
        targets: Target tensor for the loss function.
        direction: A list of tensors representing the direction vector in parameter space.
                   Must have the same structure and shape as model.parameters().
        alpha_range: A tuple (min_alpha, max_alpha) defining the range for the 1D slice.
        num_points: Number of points to sample along the 1D slice.

    Returns:
        A tuple (alphas, losses) where alphas are the scaled distances along the direction
        and losses are the corresponding loss values.
    """
    original_params = [p.data.clone() for p in model.parameters()]
    alphas = torch.linspace(alpha_range[0], alpha_range[1], num_points)
    losses = []

    with torch.no_grad():
        for alpha in alphas:
            # Update model parameters along the direction
            for p, d, original_p in zip(model.parameters(), direction, original_params):
                p.data = original_p + alpha * d

            # Compute loss
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            losses.append(loss.item())

        # Restore original parameters
        for p, original_p in zip(model.parameters(), original_params):
            p.data = original_p

    return alphas, torch.tensor(losses)


def compute_loss_surface_2d(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    direction1: List[torch.Tensor],
    direction2: List[torch.Tensor],
    alpha_range: Tuple[float, float] = (-1.0, 1.0),
    beta_range: Tuple[float, float] = (-1.0, 1.0),
    num_points: int = 25,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes loss values over a 2D plane in the parameter space.

    Args:
        model: The PyTorch model.
        loss_fn: The loss function.
        inputs: Input tensor to the model.
        targets: Target tensor for the loss function.
        direction1: First direction vector in parameter space.
        direction2: Second direction vector in parameter space.
        alpha_range: A tuple (min_alpha, max_alpha) defining the range for the first direction.
        beta_range: A tuple (min_beta, max_beta) defining the range for the second direction.
        num_points: Number of points to sample along each dimension of the 2D plane.

    Returns:
        A tuple (alphas, betas, losses_surface) where alphas and betas are the grid coordinates
        and losses_surface is a 2D tensor of corresponding loss values.
    """
    original_params = [p.data.clone() for p in model.parameters()]
    alphas = torch.linspace(alpha_range[0], alpha_range[1], num_points)
    betas = torch.linspace(beta_range[0], beta_range[1], num_points)
    losses_surface = torch.zeros(num_points, num_points)

    with torch.no_grad():
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                # Update model parameters along the two directions
                for p, d1, d2, original_p in zip(
                    model.parameters(), direction1, direction2, original_params
                ):
                    p.data = original_p + alpha * d1 + beta * d2

                # Compute loss
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                losses_surface[i, j] = loss.item()

        # Restore original parameters
        for p, original_p in zip(model.parameters(), original_params):
            p.data = original_p

    return alphas, betas, losses_surface


def create_random_direction(
    model: nn.Module,
) -> List[torch.Tensor]:
    """Creates a random normalized direction vector in parameter space.

    The direction vector has the same structure and shape as the model's parameters.

    Args:
        model: The PyTorch model.

    Returns:
        A list of tensors representing the random normalized direction vector.
    """
    direction = []
    for p in model.parameters():
        d = torch.randn_like(p)
        d_norm = torch.norm(d)
        if d_norm > 0:
            direction.append(d / d_norm)
        else:
            direction.append(torch.zeros_like(p))
    return direction
