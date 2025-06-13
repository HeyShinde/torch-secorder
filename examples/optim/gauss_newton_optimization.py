"""Example demonstrating a simplified Gauss-Newton optimization step using GNM diagonal."""

import torch
import torch.nn as nn
import torch.optim as optim

from torch_secorder.approximations.gauss_newton import gauss_newton_matrix_approximation


# 1. Define a Simple Model (e.g., for non-linear least squares)
class SimpleNonLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Let's make it slightly non-linear
        self.linear1 = nn.Linear(2, 5)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(5, 1)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


# 2. Generate Synthetic Data
torch.manual_seed(42)
X = torch.randn(100, 2) * 5  # Input features
y = (X[:, 0] ** 2 + X[:, 1] * 3 + 2 + torch.randn(100) * 0.5).unsqueeze(
    1
)  # Non-linear target

# 3. Instantiate Model, Loss
model = SimpleNonLinearModel()
loss_fn = nn.MSELoss()  # Gauss-Newton is for least squares

learning_rate = 0.01
num_iterations = 100

print("\n--- Training with Approximate Gauss-Newton (Diagonal GNM) ---")
for iteration in range(num_iterations):
    model.train()
    optimizer = optim.SGD(
        model.parameters(), lr=learning_rate
    )  # We'll manually adjust gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X)
    loss = loss_fn(outputs, y)

    # Backward pass for gradients
    loss.backward(
        create_graph=True
    )  # create_graph=True if we want higher-order, not strictly needed for GNM diag

    # Compute Diagonal of Gauss-Newton Matrix
    # The inverse of the GNM is used in the update rule: delta_theta = - G^{-1} * gradient
    # Here, we use diagonal GNM, so G^{-1}_ii = 1 / G_ii
    gnm_diagonal_list = list(gauss_newton_matrix_approximation(model, loss_fn, X, y))

    # Manual Gauss-Newton Update (approximated with diagonal GNM)
    with torch.no_grad():
        for i, param in enumerate(model.parameters()):
            if param.grad is not None:
                # Update: param.data -= learning_rate * (param.grad / (gnm_diag + epsilon))
                epsilon = 1e-6  # For numerical stability
                # Ensure gnm_diagonal_list[i] has the same shape as param.grad
                param.grad.data.div_(
                    gnm_diagonal_list[i].clamp(min=epsilon)
                )  # Element-wise division

        optimizer.step()  # Apply the modified gradients

    if (iteration + 1) % 10 == 0:
        print(f"Iteration [{iteration + 1}/{num_iterations}], Loss: {loss.item():.4f}")

print("\nTraining complete.")
print(f"Final Loss: {loss.item():.4f}")
print("Model parameters after Gauss-Newton approximation:")
for name, param in model.named_parameters():
    print(f"{name}: {param.data.squeeze()}")
