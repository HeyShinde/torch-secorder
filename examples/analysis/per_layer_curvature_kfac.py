"""Example demonstrating per-layer curvature extraction and its relation to K-FAC.

This script shows how to extract per-layer Fisher diagonal elements using per_layer_fisher_diagonal.
These per-layer blocks are the foundation for Kronecker-Factored Approximate Curvature (K-FAC).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_secorder.core.per_layer_curvature import per_layer_fisher_diagonal


# Define a simple model with multiple layers
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 3)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)


# Generate synthetic data
torch.manual_seed(42)
X = torch.randn(8, 10)
y = torch.randint(0, 3, (8,))  # 3 classes

# Instantiate the model
model = SimpleModel()

# Compute per-layer Fisher diagonal
# This extracts block-diagonal approximations of the Fisher Information Matrix
# These blocks are the basis for K-FAC approximations
layer_fisher_diagonals = per_layer_fisher_diagonal(model, F.cross_entropy, X, y)

print("Per-Layer Fisher Diagonal (K-FAC basis):")
for layer_name, diag in layer_fisher_diagonals.items():
    print(
        f"Layer: {layer_name}, Shape: {diag.shape}, First few values: {diag.flatten()[:5].tolist()}"
    )

print(
    "\nK-FAC would further decompose these blocks into Kronecker products of smaller matrices."
)
