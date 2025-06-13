"""Example demonstrating Natural Gradient Descent using Fisher diagonal approximation."""

import torch
import torch.nn as nn
import torch.optim as optim

from torch_secorder.approximations.fisher_diagonal import empirical_fisher_diagonal


# 1. Define a Simple Model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


# 2. Generate Synthetic Data
torch.manual_seed(42)
X = torch.randn(100, 2)
y = X @ torch.tensor([[1.0], [2.0]]) + 0.5 + torch.randn(100, 1) * 0.1

# 3. Instantiate Model, Loss, and Optimizer
model = SimpleModel()
loss_fn = nn.MSELoss()

# In a real NGD optimizer, this would be integrated directly.
# Here, we demonstrate the manual step using Fisher diagonal.

learning_rate = 0.01
num_epochs = 100

print("\n--- Training with Natural Gradient Descent (Approximation) ---")
for epoch in range(num_epochs):
    model.train()
    optimizer = optim.SGD(
        model.parameters(), lr=learning_rate
    )  # We'll manually adjust gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X)
    loss = loss_fn(outputs, y)

    # Backward pass to compute gradients
    loss.backward(
        create_graph=True
    )  # create_graph=True is needed for higher-order derivatives like Fisher

    # Compute Empirical Fisher Diagonal
    # For NGD, we need F^{-1}g. Here, we approximate F as diagonal.
    fisher_diagonal_list = empirical_fisher_diagonal(model, loss_fn, X, y)

    # Manual Natural Gradient Update (approximated with diagonal Fisher)
    with torch.no_grad():
        for i, param in enumerate(model.parameters()):
            if param.grad is not None:
                # Natural gradient: F^{-1} * gradient
                # If F is diagonal, F_ii = fisher_diagonal_list[i]_ii
                # So F^{-1}_ii = 1 / fisher_diagonal_list[i]_ii
                # Natural gradient update for each parameter:
                # param_update = learning_rate * (param.grad / fisher_diagonal_list[i])
                # param.data -= param_update
                # A simpler diagonal update often used in practice for NGD is param.grad / (F_diag + epsilon)

                # Let's use a simpler diagonal update: param.grad / (fisher_diag + epsilon)
                epsilon = 1e-6  # For numerical stability
                param.grad.data.div_(
                    fisher_diagonal_list[i].clamp(min=epsilon)
                )  # Element-wise division

        optimizer.step()  # Apply the modified gradients

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

print("\nTraining complete.")
print(f"Final Loss: {loss.item():.4f}")
print("Model parameters after NGD:")
for name, param in model.named_parameters():
    print(f"{name}: {param.data.squeeze()}")
