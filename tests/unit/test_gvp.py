import torch
import torch.nn as nn

from torch_secorder.core.gvp import (
    batch_jvp,
    batch_vjp,
    full_jacobian,
    jvp,
    model_jvp,
    model_vjp,
    vjp,
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


def test_jvp_scalar_output():
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    v = torch.tensor([0.5, -1.0])

    def func():
        return x[0] ** 2 + 3 * x[1] ** 2

    # Analytical JVP: [2*x[0], 6*x[1]] @ v
    expected = 2 * x[0] * v[0] + 6 * x[1] * v[1]
    result = jvp(func, [x], v)
    assert torch.allclose(result, expected)


def test_jvp_vector_output():
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    v = torch.tensor([0.5, -1.0])

    def func():
        return torch.stack([x[0] ** 2, 3 * x[1] ** 2])

    # Analytical JVP: [2*x[0]*v[0], 6*x[1]*v[1]]
    expected = torch.tensor([2 * x[0] * v[0], 6 * x[1] * v[1]])
    result = jvp(func, [x], v)
    assert torch.allclose(result, expected)


def test_vjp_scalar_output():
    x = torch.tensor([1.0, 2.0], requires_grad=True)

    def func():
        return x[0] ** 2 + 3 * x[1] ** 2

    v = torch.tensor(2.0)  # scalar output, so v is scalar
    # Analytical VJP: v * grad = v * [2*x[0], 6*x[1]]
    expected = torch.tensor([2 * x[0] * v, 6 * x[1] * v])
    result = vjp(func, [x], v)
    assert torch.allclose(result, expected)


def test_vjp_vector_output():
    x = torch.tensor([1.0, 2.0], requires_grad=True)

    def func():
        return torch.stack([x[0] ** 2, 3 * x[1] ** 2])

    v = torch.tensor([2.0, 3.0])
    # Analytical VJP: [2*v[0]*x[0], 6*v[1]*x[1]]
    expected = torch.tensor([2 * v[0] * x[0], 6 * v[1] * x[1]])
    result = vjp(func, [x], v)
    assert torch.allclose(result, expected)


def test_model_jvp():
    model = SimpleModel()
    x = torch.randn(1, 2)
    v = [torch.randn_like(p) for p in model.parameters()]
    # Just check output shape matches model(x)
    result = model_jvp(model, x, v)
    assert result.shape == model(x).shape


def test_model_vjp():
    model = SimpleModel()
    x = torch.randn(1, 2)
    y = model(x)
    v = torch.randn_like(y)
    result = model_vjp(model, x, v)
    params = list(model.parameters())
    assert isinstance(result, list)
    assert len(result) == len(params)
    for r, p in zip(result, params):
        assert r.shape == p.shape


def test_batch_jvp():
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    vs = torch.stack([torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])])

    def func():
        return torch.stack([x[0] ** 2, 3 * x[1] ** 2])

    result = batch_jvp(func, [x], vs)
    # Analytical: [[2*x[0], 0], [0, 6*x[1]]]
    expected = torch.stack(
        [torch.tensor([2 * x[0], 0.0]), torch.tensor([0.0, 6 * x[1]])]
    )
    assert torch.allclose(result, expected)


def test_batch_vjp():
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    vs = torch.stack([torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])])

    def func():
        return torch.stack([x[0] ** 2, 3 * x[1] ** 2])

    result = batch_vjp(func, [x], vs)
    # Analytical: [[2*x[0], 0], [0, 6*x[1]]]
    expected = torch.stack(
        [torch.tensor([2 * x[0], 0.0]), torch.tensor([0.0, 6 * x[1]])]
    )
    # result is a list of one tensor (since one param)
    assert torch.allclose(result[0], expected)


def test_full_jacobian():
    x = torch.tensor([1.0, 2.0], requires_grad=True)

    def func():
        return torch.stack([x[0] ** 2, 3 * x[1] ** 2])

    jac = full_jacobian(func, [x])
    # Should be [ [2*x[0], 0], [0, 6*x[1]] ]
    expected = torch.tensor([[2 * x[0], 0.0], [0.0, 6 * x[1]]])
    assert torch.allclose(jac[0], expected)
