"""Utility functions for handling model parameters."""

from typing import Dict, Iterable, List, Tuple, Union

import torch
from torch.nn import Module


def flatten_params(params: Iterable[torch.Tensor]) -> torch.Tensor:
    """Flattens a list of parameter tensors into a single concatenated tensor.

    Args:
        params: An iterable of PyTorch parameter tensors.

    Returns:
        A single 1D tensor containing all flattened parameters.
    """
    return torch.cat([p.view(-1) for p in params])


def unflatten_params(
    flat_params: torch.Tensor, param_shapes: List[torch.Size]
) -> List[torch.Tensor]:
    """Unflattens a single tensor of parameters back into a list of tensors with original shapes.

    Args:
        flat_params: A 1D tensor containing all flattened parameters.
        param_shapes: A list of `torch.Size` objects, representing the original shapes of the parameters.

    Returns:
        A list of PyTorch tensors with their original shapes.
    """
    unflattened_params = []
    offset = 0
    for shape in param_shapes:
        num_elements = shape.numel()
        param = flat_params[offset : offset + num_elements].view(shape)
        unflattened_params.append(param)
        offset += num_elements
    return unflattened_params


def get_param_shapes(params: Iterable[torch.Tensor]) -> List[torch.Size]:
    """Retrieves the shapes of an iterable of parameter tensors.

    Args:
        params: An iterable of PyTorch parameter tensors.

    Returns:
        A list of `torch.Size` objects, each representing the shape of a parameter.
    """
    return [p.shape for p in params]


def get_params_by_module_type(
    model: Module, module_type: Union[type, List[type], Tuple[type, ...]]
) -> Dict[str, List[torch.Tensor]]:
    """Extracts parameters belonging to specific module types.

    Args:
        model: The PyTorch model to inspect.
        module_type: The type(s) of `torch.nn.Module` to filter parameters by.
                    Can be a single type (e.g., `torch.nn.Linear`) or a list/tuple of types.

    Returns:
        Dictionary mapping module names to lists of their parameter tensors.
    """
    if not isinstance(model, Module):
        raise TypeError("model must be a torch.nn.Module")
    if isinstance(module_type, list):
        module_type = tuple(module_type)
    elif not isinstance(module_type, tuple):
        module_type = (module_type,)

    params_by_module = {}
    for name, module in model.named_modules():
        if isinstance(module, module_type):
            params_by_module[name] = list(module.parameters())  # type: ignore[attr-defined]
    return params_by_module


def get_params_by_name_pattern(model: Module, pattern: str) -> List[torch.Tensor]:
    """Extracts parameters whose names match a given pattern.

    This is useful for selecting parameters based on their hierarchical names
    within the model (e.g., "layer1.0.weight").

    Args:
        model: The PyTorch model to inspect.
        pattern: A regex pattern string to match against parameter names.

    Returns:
        A list of parameter tensors whose names match the pattern.
    """
    if not isinstance(model, Module):
        raise TypeError("model must be a torch.nn.Module")
    import re

    params = []
    for name, param in model.named_parameters():
        if re.search(pattern, name):
            params.append(param)
    return params
