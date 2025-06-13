Utility Functions
=================

.. currentmodule:: torch_secorder.core.utils

This module provides general utility functions for handling PyTorch model parameters and other common tasks in second-order computations.

Flatten Parameters
------------------

.. autofunction:: flatten_params

Example:
~~~~~~~~

.. code-block:: python

    import torch
    from torch_secorder.core.utils import flatten_params

    param1 = torch.randn(2, 3)
    param2 = torch.randn(5)
    params = [param1, param2]

    flat_params = flatten_params(params)
    print(f"Flattened parameters shape: {flat_params.shape}")

Unflatten Parameters
--------------------

.. autofunction:: unflatten_params

Example:
~~~~~~~~

.. code-block:: python

    import torch
    from torch_secorder.core.utils import unflatten_params, get_param_shapes

    param1 = torch.randn(2, 3)
    param2 = torch.randn(5)
    params = [param1, param2]

    flat_params = flatten_params(params)
    param_shapes = get_param_shapes(params)

    unflat_params = unflatten_params(flat_params, param_shapes)
    print(f"Unflattened param1 shape: {unflat_params[0].shape}")
    print(f"Unflattened param2 shape: {unflat_params[1].shape}")

Get Parameter Shapes
--------------------

.. autofunction:: get_param_shapes

Example:
~~~~~~~~

.. code-block:: python

    import torch
    from torch_secorder.core.utils import get_param_shapes

    param1 = torch.randn(2, 3)
    param2 = torch.randn(5)
    params = [param1, param2]

    shapes = get_param_shapes(params)
    print(f"Parameter shapes: {shapes}")

Get Parameters by Module Type
-----------------------------

.. autofunction:: get_params_by_module_type

Example:
~~~~~~~~

.. code-block:: python

    import torch.nn as nn
    from torch_secorder.core.utils import get_params_by_module_type

    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 5)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(5, 1)
            self.conv_layer = nn.Conv2d(3, 16, 3) # Add a conv layer for multi-type example

        def forward(self, x):
            return self.linear2(self.relu(self.linear1(x)))

    model = MyModel()

    # Get parameters from Linear layers
    linear_params_dict = get_params_by_module_type(model, nn.Linear)
    print(f"Parameters from Linear layers (dict keys): {list(linear_params_dict.keys())}")
    # Example of accessing and printing shapes
    for name, params_list in linear_params_dict.items():
        print(f"  {name}: {[p.shape for p in params_list]}")

    # Get parameters from both Linear and Conv2d layers
    linear_and_conv_params_dict = get_params_by_module_type(model, [nn.Linear, nn.Conv2d])
    print(f"\nParameters from Linear and Conv2d layers (dict keys): {list(linear_and_conv_params_dict.keys())}")
    for name, params_list in linear_and_conv_params_dict.items():
        print(f"  {name}: {[p.shape for p in params_list]}")

Get Parameters by Name Pattern
------------------------------

.. autofunction:: get_params_by_name_pattern

Example:
~~~~~~~~

.. code-block:: python

    import torch.nn as nn
    from torch_secorder.core.utils import get_params_by_name_pattern

    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 5)
            self.layer2 = nn.Linear(5, 1)

        def forward(self, x):
            return self.layer2(self.layer1(x))

    model = MyModel()
    weight_params = get_params_by_name_pattern(model, "weight")
    print(f"Weight parameters: {[p.shape for p in weight_params]}")

    layer1_params = get_params_by_name_pattern(model, "layer1")
    print(f"Layer1 parameters: {[p.shape for p in layer1_params]}")

Notes
-----

1.  **Parameter Handling**: These utilities are crucial for algorithms that operate on the entire parameter vector of a neural network, such as some second-order optimization methods or curvature analysis techniques.

2.  **`flatten_params`**: This function concatenates all parameter tensors into a single 1D tensor. This simplifies operations that treat all parameters as a single, contiguous block of memory.

3.  **`unflatten_params`**: This function reconstructs the original parameter tensors from a flattened vector, given their original shapes. It's essential for converting back to the model's parameter structure after performing operations on the flattened vector.

4.  **`get_param_shapes`**: This helper function provides the necessary shape information for `unflatten_params` to correctly reconstruct the original tensors.

5.  **`get_params_by_module_type`**: This function allows targeted extraction of parameters based on the type of PyTorch module they belong to (e.g., all parameters from `nn.Linear` layers). This is useful for layer-wise analysis or optimization.

6.  **`get_params_by_name_pattern`**: This function enables filtering parameters by their names using regular expressions. This provides flexible control over selecting parameters from specific parts of the model hierarchy.
