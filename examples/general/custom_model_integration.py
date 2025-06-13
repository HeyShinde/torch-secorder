"""Example demonstrating integration with custom PyTorch models.

This example shows how to use torch-secorder with custom PyTorch models,
including models with:
1. Custom forward passes
2. Multiple outputs
3. Complex loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_secorder.core import gvp, hessian_diagonal, hvp


class CustomModel(nn.Module):
    """A custom model with multiple outputs and a complex forward pass."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))
        self.classifier = nn.Linear(10, 2)
        self.regressor = nn.Linear(10, 1)

    def forward(self, x):
        features = self.encoder(x)
        return {
            "logits": self.classifier(features),
            "predictions": self.regressor(features),
        }


def custom_loss(outputs, targets):
    """A custom loss function combining classification and regression losses."""
    classification_loss = F.cross_entropy(outputs["logits"], targets["classes"])
    regression_loss = F.mse_loss(outputs["predictions"], targets["values"])
    return classification_loss + 0.1 * regression_loss


def main():
    # 1. Create the custom model
    model = CustomModel()

    # 2. Generate random data
    x = torch.randn(32, 10)
    y = {"classes": torch.randint(0, 2, (32,)), "values": torch.randn(32, 1)}

    # 3. Compute loss
    def get_loss():
        return custom_loss(model(x), y)

    # 4. Create a random direction vector
    v = [torch.randn_like(p) for p in model.parameters()]

    # 5. Compute HVP with custom model and loss
    hvp_result = hvp.model_hvp(model, custom_loss, x, y, v)
    print("HVP computed successfully with custom model!")

    # 6. Compute JVP for each output separately
    # For models with multiple outputs, we need to compute JVP for each output tensor
    # We'll create wrapper models for each output

    class LogitsModel(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, x):
            return self.base_model(x)["logits"]

    class PredictionsModel(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, x):
            return self.base_model(x)["predictions"]

    # Create wrapper models
    logits_model = LogitsModel(model)
    predictions_model = PredictionsModel(model)

    # Compute JVPs using the wrapper models
    jvp_logits = gvp.model_jvp(logits_model, x, v)
    print("JVP computed successfully for logits!")

    jvp_predictions = gvp.model_jvp(predictions_model, x, v)
    print("JVP computed successfully for predictions!")

    # 7. Compute Hessian diagonal
    hessian_diag = hessian_diagonal(get_loss, list(model.parameters()))
    print("Hessian diagonal computed successfully with custom model!")

    # 8. Print shapes for verification
    print("\nShapes of results:")
    print(f"HVP: {[h.shape for h in hvp_result]}")
    print(f"JVP (logits): {[j.shape for j in jvp_logits]}")
    print(f"JVP (predictions): {[j.shape for j in jvp_predictions]}")
    print(f"Hessian diagonal: {[h.shape for h in hessian_diag]}")


if __name__ == "__main__":
    main()
