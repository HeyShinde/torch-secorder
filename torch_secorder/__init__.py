"""Torch-Secorder: A PyTorch library for second-order optimization and analysis."""

from torch_secorder.analysis import (
    compute_loss_surface_1d,
    compute_loss_surface_2d,
    estimate_eigenvalues,
    lanczos_iteration,
    power_iteration,
)
from torch_secorder.approximations import (
    empirical_fisher_diagonal,
    empirical_fisher_trace,
    gauss_newton_matrix_approximation,
    generalized_fisher_diagonal,
    generalized_fisher_trace,
)
from torch_secorder.core import (
    approximate_hvp,
    batch_jvp,
    batch_vjp,
    exact_hvp,
    flatten_params,
    full_jacobian,
    gauss_newton_product,
    get_layer_curvature_stats,
    get_param_shapes,
    get_params_by_module_type,
    get_params_by_name_pattern,
    hessian_diagonal,
    hessian_trace,
    jvp,
    model_hessian_diagonal,
    model_hessian_trace,
    model_hvp,
    model_jvp,
    model_vjp,
    per_layer_fisher_diagonal,
    per_layer_hessian_diagonal,
    unflatten_params,
    vjp,
)

__version__ = "0.0.1"

__all__ = [
    # Core HVP
    "exact_hvp",
    "approximate_hvp",
    "model_hvp",
    "gauss_newton_product",
    # Core GVP
    "jvp",
    "vjp",
    "model_jvp",
    "model_vjp",
    "batch_jvp",
    "batch_vjp",
    "full_jacobian",
    # Core Hessian
    "hessian_diagonal",
    "model_hessian_diagonal",
    "hessian_trace",
    "model_hessian_trace",
    # Core Utils
    "flatten_params",
    "unflatten_params",
    "get_param_shapes",
    "get_params_by_module_type",
    "get_params_by_name_pattern",
    # Core Per-layer
    "per_layer_hessian_diagonal",
    "per_layer_fisher_diagonal",
    "get_layer_curvature_stats",
    # Approximations
    "empirical_fisher_diagonal",
    "generalized_fisher_diagonal",
    "empirical_fisher_trace",
    "generalized_fisher_trace",
    "gauss_newton_matrix_approximation",
    # Analysis
    "compute_loss_surface_1d",
    "compute_loss_surface_2d",
    "power_iteration",
    "lanczos_iteration",
    "estimate_eigenvalues",
]
