Practical Applications & Tutorials
==================================

This section provides detailed examples of using `torch-secorder` in various scenarios. For basic usage of foundational tools like Hessian-Vector Products (HVPs), Jacobian-Vector Products (JVPs), and basic curvature statistics (Hessian diagonal/trace), please refer to their respective API documentation in the `Core` and `Approximations` modules.

Analysis Examples
-----------------

Loss Landscape Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example demonstrates how to compute and visualize the loss landscape of a neural network model. The loss landscape provides insights into the geometry of the optimization space and can help understand:

- The smoothness and convexity of the loss function
- The presence of local minima and saddle points
- The difficulty of optimization in different regions

The example shows two types of visualizations:

1. **1D Loss Surface**: A slice of the loss landscape along a random direction, showing how the loss changes as we move away from the current parameter point.

2. **2D Loss Surface**: A contour plot showing the loss landscape in a 2D plane defined by two random directions, providing a more comprehensive view of the optimization space.

The visualization generates two files:
- `1d_loss_surface.png`: A line plot showing the loss along a single direction
- `2d_loss_surface.png`: A 3D surface plot showing the loss in a 2D plane

.. literalinclude:: ../examples/analysis/loss_landscape_visualization.py
   :language: python
   :linenos:

Per-Layer Curvature Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example demonstrates how to extract per-layer Fisher diagonal elements, which form the basis for Kronecker-Factored Approximate Curvature (K-FAC) methods.

.. literalinclude:: ../examples/analysis/per_layer_curvature_kfac.py
   :language: python
   :linenos:

Optimization Examples
---------------------

Natural Gradient Descent
~~~~~~~~~~~~~~~~~~~~~~~~

This example demonstrates a simplified implementation of Natural Gradient Descent, leveraging the Fisher diagonal approximation from `torch-secorder`.

.. literalinclude:: ../examples/optim/natural_gradient_descent.py
   :language: python
   :linenos:

Bayesian Neural Networks Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example illustrates how Fisher Information can be used in the context of Bayesian Neural Networks to approximate posterior variances of parameters.

.. literalinclude:: ../examples/optim/bayesian_neural_network_inference.py
   :language: python
   :linenos:

Gauss-Newton Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~

This example demonstrates a simplified optimization loop using the diagonal Gauss-Newton Matrix approximation.

.. literalinclude:: ../examples/optim/gauss_newton_optimization.py
   :language: python
   :linenos:

General Examples
----------------

Quick Start
~~~~~~~~~~~

A minimal example showing how to get started with `torch-secorder`.

.. literalinclude:: ../examples/general/quick_start.py
   :language: python
   :linenos:

Custom Model Integration
~~~~~~~~~~~~~~~~~~~~~~~~

Demonstrates how to integrate `torch-secorder` with custom PyTorch models.

.. literalinclude:: ../examples/general/custom_model_integration.py
   :language: python
   :linenos:
