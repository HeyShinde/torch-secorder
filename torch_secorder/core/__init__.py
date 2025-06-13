"""Core second-order operations module."""

from torch_secorder.core.gvp import (
    batch_jvp,
    batch_vjp,
    full_jacobian,
    jvp,
    model_jvp,
    model_vjp,
    vjp,
)
from torch_secorder.core.hessian_diagonal import (
    hessian_diagonal,
    model_hessian_diagonal,
)
from torch_secorder.core.hessian_trace import hessian_trace, model_hessian_trace
from torch_secorder.core.hvp import (
    approximate_hvp,
    exact_hvp,
    gauss_newton_product,
    model_hvp,
)
from torch_secorder.core.per_layer_curvature import (
    get_layer_curvature_stats,
    per_layer_fisher_diagonal,
    per_layer_hessian_diagonal,
)
from torch_secorder.core.utils import (
    flatten_params,
    get_param_shapes,
    get_params_by_module_type,
    get_params_by_name_pattern,
    unflatten_params,
)

__all__ = [
    # HVP
    "exact_hvp",
    "approximate_hvp",
    "model_hvp",
    "gauss_newton_product",
    # GVP
    "jvp",
    "vjp",
    "model_jvp",
    "model_vjp",
    "batch_jvp",
    "batch_vjp",
    "full_jacobian",
    # Hessian Diagonal
    "hessian_diagonal",
    "model_hessian_diagonal",
    # Hessian Trace
    "hessian_trace",
    "model_hessian_trace",
    # Utils
    "flatten_params",
    "unflatten_params",
    "get_param_shapes",
    "get_params_by_module_type",
    "get_params_by_name_pattern",
    # Per-layer curvature
    "per_layer_hessian_diagonal",
    "per_layer_fisher_diagonal",
    "get_layer_curvature_stats",
]
