"""Quick start example for torch-secorder.

This example demonstrates basic usage of torch-secorder's core functionality:
1. Computing Hessian-Vector Products (HVPs)
2. Computing Jacobian-Vector Products (JVPs)
3. Computing Hessian diagonal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_secorder.core import gvp, hessian_diagonal, hvp


def main():
    # 1. Create a simple model
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))

    # 2. Generate some random data
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)

    # 3. Compute loss
    def get_loss():
        return F.mse_loss(model(x), y)

    # 4. Create a random direction vector
    v = [torch.randn_like(p) for p in model.parameters()]

    # 5. Compute HVP
    hvp_result = hvp.model_hvp(model, F.mse_loss, x, y, v)
    print("HVP computed successfully!")

    # 6. Compute JVP
    jvp_result = gvp.model_jvp(model, x, v)
    print("JVP computed successfully!")

    # 7. Compute Hessian diagonal
    hessian_diag = hessian_diagonal(get_loss, list(model.parameters()))
    print("Hessian diagonal computed successfully!")

    # 8. Print shapes for verification
    print("\nShapes of results:")
    print(f"HVP: {[h.shape for h in hvp_result]}")
    print(f"JVP: {[j.shape for j in jvp_result]}")
    print(f"Hessian diagonal: {[h.shape for h in hessian_diag]}")


if __name__ == "__main__":
    main()
