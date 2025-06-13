"""Tests for the utility functions in torch_secorder.core.utils."""

import torch
import torch.nn as nn

from torch_secorder.core.utils import (
    flatten_params,
    get_param_shapes,
    get_params_by_module_type,
    get_params_by_name_pattern,
    unflatten_params,
)


def test_flatten_params():
    """Test flattening of parameter tensors."""
    param1 = torch.randn(2, 3)
    param2 = torch.randn(5)
    param3 = torch.randn(1, 4)
    params = [param1, param2, param3]

    flat_params = flatten_params(params)

    expected_flat_size = param1.numel() + param2.numel() + param3.numel()
    assert flat_params.shape == (expected_flat_size,)

    # Check if the flattened tensor contains the correct values
    expected_flat = torch.cat([p.view(-1) for p in params])
    assert torch.allclose(flat_params, expected_flat)


def test_get_param_shapes():
    """Test retrieval of parameter shapes."""
    param1 = torch.randn(2, 3)
    param2 = torch.randn(5)
    param3 = torch.randn(1, 4)
    params = [param1, param2, param3]

    shapes = get_param_shapes(params)

    expected_shapes = [torch.Size([2, 3]), torch.Size([5]), torch.Size([1, 4])]
    assert shapes == expected_shapes


def test_unflatten_params():
    """Test unflattening of parameter tensors."""
    param1 = torch.randn(2, 3)
    param2 = torch.randn(5)
    param3 = torch.randn(1, 4)
    params = [param1, param2, param3]

    flat_params = flatten_params(params)
    param_shapes = get_param_shapes(params)

    unflattened_params = unflatten_params(flat_params, param_shapes)

    assert len(unflattened_params) == len(params)
    for i in range(len(params)):
        assert unflattened_params[i].shape == params[i].shape
        assert torch.allclose(unflattened_params[i], params[i])


def test_flatten_unflatten_consistency():
    """Test consistency between flattening and unflattening."""
    param1 = torch.randn(10, 20)
    param2 = torch.randn(5)
    param3 = torch.randn(1, 2, 3)
    params = [param1, param2, param3]

    flat_params = flatten_params(params)
    param_shapes = get_param_shapes(params)
    unflattened_params = unflatten_params(flat_params, param_shapes)

    for original_p, unflattened_p in zip(params, unflattened_params):
        assert torch.allclose(original_p, unflattened_p)
        assert original_p.shape == unflattened_p.shape


def test_get_params_by_module_type():
    """Test extracting parameters by module type."""

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 5)
            self.conv1 = nn.Conv2d(3, 16, 3)
            self.linear2 = nn.Linear(5, 1)

    model = SimpleModel()
    linear_params = get_params_by_module_type(model, nn.Linear)
    conv_params = get_params_by_module_type(model, nn.Conv2d)

    # Check that all linear layer parameters are captured
    total_linear_params = sum(len(p_list) for p_list in linear_params.values())
    assert total_linear_params == 4  # 2 weights + 2 biases from two linear layers
    assert any(
        torch.equal(p, model.linear1.weight)
        for p_list in linear_params.values()
        for p in p_list
    )
    assert any(
        torch.equal(p, model.linear1.bias)
        for p_list in linear_params.values()
        for p in p_list
    )
    assert any(
        torch.equal(p, model.linear2.weight)
        for p_list in linear_params.values()
        for p in p_list
    )
    assert any(
        torch.equal(p, model.linear2.bias)
        for p_list in linear_params.values()
        for p in p_list
    )

    # Check that conv layer parameters are captured
    total_conv_params = sum(len(p_list) for p_list in conv_params.values())
    assert total_conv_params == 2  # 1 weight + 1 bias from one conv layer
    assert any(
        torch.equal(p, model.conv1.weight)
        for p_list in conv_params.values()
        for p in p_list
    )
    assert any(
        torch.equal(p, model.conv1.bias)
        for p_list in conv_params.values()
        for p in p_list
    )


def test_get_params_by_name_pattern():
    """Test extracting parameters by name pattern."""

    class NestedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 5)
            self.sub_module = nn.Sequential(nn.Linear(5, 3), nn.ReLU())
            self.output_layer = nn.Linear(3, 1)

        def forward(self, x):
            return self.output_layer(self.sub_module(self.layer1(x)))

    model = NestedModel()

    # Test with a simple pattern (e.g., 'weight')
    weight_params = get_params_by_name_pattern(model, "weight")
    assert (
        len(weight_params) == 3
    )  # layer1.weight, sub_module.0.weight, output_layer.weight
    assert any(torch.equal(p, model.layer1.weight) for p in weight_params)
    assert any(torch.equal(p, model.sub_module[0].weight) for p in weight_params)
    assert any(torch.equal(p, model.output_layer.weight) for p in weight_params)

    # Test with a more specific pattern (e.g., 'layer1')
    layer1_params = get_params_by_name_pattern(model, "layer1")
    assert len(layer1_params) == 2  # layer1.weight, layer1.bias
    assert any(torch.equal(p, model.layer1.weight) for p in layer1_params)
    assert any(torch.equal(p, model.layer1.bias) for p in layer1_params)

    # Test with a pattern for nested modules
    sub_module_params = get_params_by_name_pattern(model, "sub_module")
    assert len(sub_module_params) == 2  # sub_module.0.weight, sub_module.0.bias
    assert any(torch.equal(p, model.sub_module[0].weight) for p in sub_module_params)
    assert any(torch.equal(p, model.sub_module[0].bias) for p in sub_module_params)
