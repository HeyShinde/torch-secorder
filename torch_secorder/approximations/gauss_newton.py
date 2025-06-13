"""Functions for computing Gauss-Newton matrix approximations."""

from typing import Callable, Iterable

import torch
from torch import Tensor
from torch.nn import Module


def gauss_newton_matrix_approximation(
    model: Module,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    inputs: Tensor,
    targets: Tensor,
    create_graph: bool = False,
) -> Iterable[Tensor]:
    r"""
    Computes the diagonal approximation of the Gauss-Newton Matrix (GNM).

    The Gauss-Newton Matrix (GNM) is an approximation of the Hessian, commonly used
    in non-linear least squares problems. For a loss function defined as:

    .. math::
        L = \frac{1}{2} \|f(x; \theta) - y\|^2,

    where :math:`f(x; \theta)` is the model output, the Gauss-Newton Matrix is given by:

    .. math::
        G = J^\top J,

    where :math:`J` is the Jacobian of :math:`f(x; \theta)` with respect to the parameters :math:`\theta`.

    This function returns the diagonal elements of the GNM for each parameter.
    It achieves this by computing the sum of squared Jacobian rows corresponding to each parameter.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model.
    loss_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        The loss function, typically :class:`torch.nn.MSELoss` or :func:`torch.nn.functional.mse_loss`.
    inputs : torch.Tensor
        Input tensor to the model.
    targets : torch.Tensor
        Target tensor for the loss function.
    create_graph : bool, optional
        If True, constructs the computation graph for higher-order derivatives. Default is False.

    Returns
    -------
    Iterable[torch.Tensor]
        A list of tensors, where each tensor contains the diagonal elements of the GNM
        for the corresponding model parameter.

    Raises
    ------
    ValueError
        If ``loss_fn`` is not MSE-based, as GNM is defined specifically for least-squares problems.
    """
    # Verify that the loss function is MSE-based
    # This is a heuristic check; a more robust check might involve inspecting the loss_fn's type
    # or relying on user to pass appropriate loss.
    if not (
        isinstance(loss_fn, torch.nn.MSELoss)
        or (
            callable(loss_fn)
            and ("mse_loss" in str(loss_fn) or "MSELoss" in str(loss_fn.__class__))
        )
    ):
        raise ValueError(
            "Gauss-Newton Matrix approximation is typically used with MSE-based loss functions "
            "for least-squares problems. Please use `nn.MSELoss` or `F.mse_loss`."
        )

    # The residual r(x;theta) = f(x;theta) - y
    # The loss L = 1/2 * ||r(x;theta)||^2
    # The Jacobian of the loss w.r.t parameters is J_L = r(x;theta)^T * J_r
    # where J_r is the Jacobian of the residuals w.r.t parameters.
    # The GNM is J_r^T J_r

    # Compute the model outputs
    outputs = model(inputs)

    # Compute residuals: r = outputs - targets
    residuals = outputs - targets

    # Ensure residuals require grad if create_graph is True for higher-order derivatives
    if create_graph and not residuals.requires_grad:
        # This ensures the gradient computation for residuals can be differentiated again
        residuals.requires_grad_(True)

    param_gen = [p for p in model.parameters() if p.requires_grad]

    gnm_diagonal = []

    # Iterate over each output dimension to construct the Jacobian rows implicitly
    # This is effectively computing J^T J by summing outer products of gradients (for each output component)
    # For the diagonal, we only need the sum of squared gradients for each parameter.
    # J_r^T J_r = sum over output dimensions of (grad(r_i) * grad(r_i)^T)
    # The diagonal of J_r^T J_r is sum over output dimensions of (grad(r_i) .^ 2)

    # Loop through each element of the residual and compute its gradient w.r.t. parameters
    # Then sum the squares of these gradients to get the diagonal of GNM.
    # This is effectively computing diag(J_r^T J_r)
    for i in range(residuals.numel()):
        # Select a single element of the residual
        # Ensure the element is part of the graph for higher-order derivatives if create_graph is True
        residual_element = residuals.view(-1)[i]

        # Compute gradients of this single residual element w.r.t. all parameters that require grad
        # Use retain_graph=True because we'll call .grad multiple times in the loop
        grads_for_element = torch.autograd.grad(
            residual_element,
            param_gen,  # Use the filtered list of parameters
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=True,
        )

        if i == 0:
            # Initialize gnm_diagonal with zeros of the correct shape based on first gradients
            # Ensure these initial zeros also have requires_grad=True if create_graph is True
            gnm_diagonal = [
                torch.zeros_like(g, requires_grad=create_graph)
                for g in grads_for_element
                if g is not None
            ]

        for j, g_elem in enumerate(grads_for_element):
            if g_elem is not None:
                # Sum the squares of the gradients for each parameter
                # Ensure that g_elem.pow(2) maintains the graph if create_graph is True
                # and that the accumulation also respects it.
                if create_graph:
                    # If create_graph is True, we need to ensure g_elem has grad_fn
                    # and that its square retains grad_fn.
                    if g_elem.grad_fn is None and g_elem.requires_grad:
                        # This case implies g_elem is an input that directly requires grad
                        # or a leaf tensor. If not, it should have a grad_fn.
                        pass  # Handled by earlier create_graph=True for residual_element
                    gnm_diagonal[j] = gnm_diagonal[j] + g_elem.pow(2)
                else:
                    # If create_graph is False, simple accumulation is fine
                    gnm_diagonal[j] = gnm_diagonal[j] + g_elem.pow(2)

    # Clear the graph to avoid memory issues if create_graph was True and not needed further
    if not create_graph:
        model.zero_grad()

    return gnm_diagonal
