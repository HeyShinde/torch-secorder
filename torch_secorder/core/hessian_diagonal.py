"""Module for computing Hessian diagonal efficiently.

This module provides functions to compute the diagonal elements of the Hessian matrix
which are useful for various second-order analysis tasks.
"""

from typing import Callable, List, Optional

import torch
from torch import Tensor
from torch.nn import Module


def hessian_diagonal(
    func: Callable[[], Tensor],
    params: List[Tensor],
    v: Optional[List[Tensor]] = None,
    create_graph: bool = False,
    strict: bool = False,
) -> List[Tensor]:
    """Compute the diagonal elements of the Hessian matrix.

    This function computes the diagonal elements of the Hessian matrix by using
    double backward for each parameter element.

    Args:
        func: A callable that returns a scalar tensor (the loss).
        params: List of parameters with respect to which the Hessian is computed.
        v: Optional list of vectors to use for computing the diagonal. If None,
           computes the true diagonal. If provided, computes v_i * H_ii for each i.
        create_graph: If True, the computational graph will be constructed,
                     allowing for higher-order derivatives.
        strict: If True, an error will be raised if any parameter requires grad
                but has no gradient.

    Returns:
        List of tensors containing the diagonal elements of the Hessian for each parameter.
    """
    # Filter out parameters that don't require grad if not in strict mode
    if not strict:
        grad_params = [p for p in params if p.requires_grad]
        if not grad_params:
            # If no parameters require grad, return zeros. Ensure they are traceable if create_graph is True.
            if create_graph:
                return [(p.sum() * 0.0).expand_as(p) for p in params]
            else:
                return [torch.zeros_like(p) for p in params]
    else:
        grad_params = params
        if not all(p.requires_grad for p in params):
            raise RuntimeError(
                "One of the differentiated Tensors does not require grad"
            )

    diag_results = []
    # Compute first-order gradients. create_graph=True ensures these gradients themselves have grad_fn.
    first_grads = torch.autograd.grad(
        func(), grad_params, create_graph=True, allow_unused=True
    )

    # Create a mapping from parameter object to its first gradient for efficient lookup
    param_to_grad_map = {p: g for p, g in zip(grad_params, first_grads)}

    for p_orig_idx, p in enumerate(params):
        if p.requires_grad:
            g = param_to_grad_map.get(p)

            if g is None:
                # This parameter `p` did not receive a gradient (e.g., due to allow_unused=True).
                # Its Hessian diagonal elements will be zero. Ensure they are traceable if create_graph is True.
                if create_graph:
                    diag_results.append((p.sum() * 0.0).expand_as(p))
                else:
                    diag_results.append(torch.zeros_like(p))
                continue

            # Flatten parameter and gradient for element-wise processing
            g_flat = g.flatten()

            per_param_diagonal_elements = []
            for i in range(
                g_flat.numel()
            ):  # Iterate over flattened elements of the gradient
                # Only proceed if g_flat[i] is part of the graph and can be differentiated further
                if g_flat[i].requires_grad and g_flat[i].grad_fn is not None:
                    # Compute the gradient of the i-th element of g_flat with respect to the entire parameter p.
                    grad2 = torch.autograd.grad(
                        g_flat[i],
                        p,
                        retain_graph=True,
                        create_graph=create_graph,
                        allow_unused=True,
                    )[0]

                    if grad2 is not None:
                        diag_elem = grad2.flatten()[i]
                    else:
                        # g_flat[i] required grad and had grad_fn, but still no dependency on p.
                        # This should still yield a traceable zero.
                        if create_graph:
                            diag_elem = (p.sum() * 0.0).expand_as(g_flat[i])
                        else:
                            diag_elem = torch.zeros_like(g_flat[i])
                else:
                    # If g_flat[i] does not require grad, or has no grad_fn (i.e., it's a leaf not connected for second derivatives),
                    # its second derivative w.r.t. p is zero.
                    if create_graph:
                        diag_elem = (p.sum() * 0.0).expand_as(g_flat[i])
                    else:
                        diag_elem = torch.zeros_like(g_flat[i])

                if v is not None:
                    v_param_elem = v[p_orig_idx].flatten()[i]
                    per_param_diagonal_elements.append(diag_elem * v_param_elem)
                else:
                    per_param_diagonal_elements.append(diag_elem)

            if per_param_diagonal_elements:
                # Stack elements and reshape to original parameter shape
                # Ensure the stacked tensor also retains grad_fn if create_graph is True
                stacked_elements = torch.stack(per_param_diagonal_elements)
                diag_results.append(stacked_elements.reshape(p.shape))
            else:
                if create_graph:
                    diag_results.append((p.sum() * 0.0).expand_as(p))
                else:
                    diag_results.append(torch.zeros_like(p))

        else:
            diag_results.append(torch.zeros_like(p))

    return diag_results


def model_hessian_diagonal(
    model: Module,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    inputs: Tensor,
    targets: Tensor,
    create_graph: bool = False,
    strict: bool = False,
) -> List[Tensor]:
    def loss_func():
        outputs = model(inputs)
        return loss_fn(outputs, targets)

    return hessian_diagonal(
        loss_func,
        list(model.parameters()),
        create_graph=create_graph,
        strict=strict,
    )
