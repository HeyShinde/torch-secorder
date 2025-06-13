"""Example demonstrating the use of Fisher Information for Bayesian Neural Networks (BNNs)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_secorder.approximations import (
    empirical_fisher_diagonal,
    generalized_fisher_diagonal,
)


# 1. Define a Simple Bayesian Neural Network (for demonstration purposes)
# In a true BNN, weights would be distributions, but here we'll use point estimates
# and use Fisher to infer uncertainty or for approximate posterior.
class SimpleBNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)


# 2. Generate Synthetic Data
torch.manual_seed(42)
X_train = torch.randn(100, 10)
y_train = X_train @ torch.randn(10, 1) + 0.5 + torch.randn(100, 1) * 0.1

X_test = torch.randn(20, 10)

# 3. Instantiate Model and Loss Function
model = SimpleBNN()

# For BNNs, often working with negative log-likelihood (NLL) directly
# For regression, this might be Gaussian NLL. For classification, CrossEntropy is NLL.
# Here, we'll use MSE for simplicity and demonstrate Fisher diagonal for it.
# Note: Generalized Fisher is typically for likelihoods (e.g., NLL in classification).
# For MSE in regression, Empirical Fisher is often more directly applicable.

# Let's simulate a regression task with a Gaussian likelihood assumption
# where the loss is proportional to MSE (which is -log-likelihood for Gaussian with fixed variance)
# We'll adapt `generalized_fisher_diagonal` for a simplified regression NLL scenario.

print("\n--- Estimating Uncertainty using Fisher Information (BNN Context) ---")

# Train the model (simplified training loop)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = F.mse_loss(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/50], Loss: {loss.item():.4f}")

# 4. Use Fisher Information for Uncertainty Estimation (Approximation)
# For a BNN, the inverse Fisher (or Hessian) often approximates the posterior covariance.
# The diagonal of F^{-1} gives approximate posterior variances of parameters.

print(
    "\n--- Computing Fisher Diagonal for Approximate Posterior Variance (Regression: MSE) ---"
)

print(
    "\nUsing Empirical Fisher Diagonal for parameter uncertainty approximation (MSE regression)..."
)
try:
    # Using empirical_fisher_diagonal for a direct demonstration applicable to MSE
    fisher_diagonal_params = empirical_fisher_diagonal(
        model, F.mse_loss, X_train, y_train
    )

    print("\nApproximate Posterior Variances (from inverse Fisher diagonal):")
    with torch.no_grad():
        for i, param in enumerate(model.parameters()):
            # Inverse of Fisher diagonal approximates variance of parameters
            # Ensure no division by zero if Fisher diagonal element is too small
            variance_approx = 1.0 / (
                fisher_diagonal_params[i] + 1e-8
            )  # Add epsilon for stability
            print(
                f"Parameter {i} variance approximation shape: {variance_approx.shape}"
            )
            print(f"  First few values: {variance_approx.flatten()[:5].tolist()}")

except NotImplementedError as e:
    print(
        f"Error: {e}. Generalized Fisher diagonal for regression NLL might not be fully supported yet in `generalized_fisher_diagonal` for direct MSE loss."
    )
except Exception as e:
    print(f"An unexpected error occurred: {e}")

print("\nBNN Fisher Information example complete.")

# Add example for generalized_fisher_diagonal (classification use case)
print("\n--- Using Generalized Fisher Diagonal (Classification Example) ---")


# Create a simple classification model and data
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 3)

    def forward(self, x):
        return self.linear(x)


clf_model = SimpleClassifier()
clf_inputs = torch.randn(8, 10)
clf_targets = torch.randint(0, 3, (8,))  # 3 classes
clf_outputs = clf_model(clf_inputs)

# Compute the Generalized Fisher Diagonal (NLL loss type)
gfim_diagonal = generalized_fisher_diagonal(
    clf_model, clf_outputs, clf_targets, loss_type="nll"
)
print("Generalized Fisher Diagonal (classification):")
for i, diag in enumerate(gfim_diagonal):
    print(f"  Param {i} shape: {diag.shape}, first few: {diag.flatten()[:5].tolist()}")
