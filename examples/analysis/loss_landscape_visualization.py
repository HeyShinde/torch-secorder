"""Example demonstrating loss landscape visualization.

This script shows how to compute and visualize 1D slices and 2D contours
of the loss landscape for a simple model, using random directions.

The example demonstrates:
1. Computing 1D loss surface along a random direction
2. Computing 2D loss surface using two random directions
3. Visualizing both surfaces using matplotlib

Requirements:
    - torch
    - matplotlib
    - numpy
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import cm

from torch_secorder.analysis.landscape import (
    compute_loss_surface_1d,
    compute_loss_surface_2d,
    create_random_direction,
)


# 1. Define a Simple Model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


# 2. Generate Synthetic Data
torch.manual_seed(42)
X_train = torch.randn(20, 2)
y_train = X_train @ torch.tensor([[2.0], [3.0]]) + 1.0 + torch.randn(20, 1) * 0.5


# 3. Instantiate Model and Loss Function
model = SimpleModel()
loss_fn = F.mse_loss


print("--- Loss Landscape Visualization Example ---")

# 4. Compute 1D Loss Surface
print("\nComputing 1D loss surface...")
direction_1d = create_random_direction(model)
alphas_1d, losses_1d = compute_loss_surface_1d(
    model,
    loss_fn,
    X_train,
    y_train,
    direction_1d,
    alpha_range=(-2.0, 2.0),
    num_points=50,
)

print("1D Loss Surface (first 5 points):\nAlphas: ", alphas_1d[:5].tolist())
print("Losses: ", losses_1d[:5].tolist())

# Plot 1D Loss Surface
plt.figure(figsize=(8, 6))
plt.plot(alphas_1d.numpy(), losses_1d.numpy(), "b-", linewidth=2)
plt.xlabel("Alpha (Direction Scale)")
plt.ylabel("Loss")
plt.title("1D Loss Surface")
plt.grid(True)
plt.savefig("1d_loss_surface.png")
plt.close()


# 5. Compute 2D Loss Surface
print("\nComputing 2D loss surface...")
direction1_2d = create_random_direction(model)
direction2_2d = create_random_direction(model)

# Ensure directions are not collinear (optional, but good for meaningful 2D surface)
# A simple way to get somewhat orthogonal directions: re-randomize if dot product is too high
# This is a heuristic, proper orthogonalization methods might be preferred for robustness
if (
    torch.dot(
        torch.cat([d.flatten() for d in direction1_2d]),
        torch.cat([d.flatten() for d in direction2_2d]),
    ).abs()
    > 0.5
):
    print("Adjusting second random direction for better orthogonality...")
    direction2_2d = create_random_direction(model)

alphas_2d, betas_2d, losses_2d = compute_loss_surface_2d(
    model, loss_fn, X_train, y_train, direction1_2d, direction2_2d, num_points=25
)

print("2D Loss Surface (top-left 3x3 values):\n", losses_2d[:3, :3].tolist())

# Plot 2D Loss Surface
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")
A, B = torch.meshgrid(alphas_2d, betas_2d, indexing="ij")
surf = ax.plot_surface(
    A.numpy(),
    B.numpy(),
    losses_2d.numpy(),
    cmap=cm.viridis,
    edgecolor="none",
    alpha=0.8,
)
ax.set_xlabel("Alpha (Direction 1)")
ax.set_ylabel("Beta (Direction 2)")
ax.set_zlabel("Loss")
ax.set_title("2D Loss Surface")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig("2d_loss_surface.png")
plt.close()

print(
    "\nLoss landscape visualization complete. Check '1d_loss_surface.png' and '2d_loss_surface.png' for the plots."
)
