"""Example script demonstrating eigenvalue analysis of a neural network."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_secorder.analysis.eigensolvers import model_eigenvalues


class SimpleNet(nn.Module):
    """A simple neural network for demonstration."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create model and data
    model = SimpleNet()
    data = torch.randn(100, 2)
    target = torch.randn(100, 1)

    # Define loss function
    def loss_fn(model, data, target):
        return F.mse_loss(model(data), target)

    # Compute top-3 eigenvalues and eigenvectors using both methods
    print("Computing eigenvalues using Power Iteration...")
    power_eigenvalues, power_eigenvectors = model_eigenvalues(
        model,
        loss_fn,
        data,
        target,
        num_eigenvalues=3,
        method="power_iteration",
        num_iterations=100,
        tol=1e-6,
    )

    print("\nComputing eigenvalues using Lanczos...")
    lanczos_eigenvalues, lanczos_eigenvectors = model_eigenvalues(
        model,
        loss_fn,
        data,
        target,
        num_eigenvalues=3,
        method="lanczos",
        num_iterations=100,
        tol=1e-6,
    )

    # Print results
    print("\nResults:")
    print("--------")
    print("Power Iteration eigenvalues:", power_eigenvalues)
    print("Lanczos eigenvalues:", lanczos_eigenvalues)

    # Compare the methods
    print("\nComparison:")
    print("-----------")
    print("Eigenvalue difference:", torch.abs(power_eigenvalues - lanczos_eigenvalues))

    # Analyze eigenvector structure
    print("\nEigenvector Analysis:")
    print("-------------------")
    for i in range(3):
        print(f"\nEigenvector {i + 1}:")
        for param, eigenvector in zip(model.parameters(), power_eigenvectors[i]):
            print(f"Parameter shape: {param.shape}")
            print(f"Eigenvector norm: {torch.norm(eigenvector)}")
            print(f"Eigenvector mean: {torch.mean(eigenvector)}")
            print(f"Eigenvector std: {torch.std(eigenvector)}")


if __name__ == "__main__":
    main()
