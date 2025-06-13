.. _landscape_api:

Loss Landscape Visualization
============================

.. currentmodule:: torch_secorder.analysis.landscape

This module provides tools for visualizing the loss landscape of neural networks, allowing researchers and practitioners to gain insights into optimization dynamics, generalization properties, and the geometry of the loss surface.

Compute 1D Loss Surface
-----------------------

.. autofunction:: compute_loss_surface_1d

Compute 2D Loss Surface
-----------------------

.. autofunction:: compute_loss_surface_2d

Create Random Direction
-----------------------

.. autofunction:: create_random_direction

Example
-------

.. literalinclude:: ../../../examples/analysis/loss_landscape_visualization.py
   :language: python
   :linenos:

Notes
-----

1.  **Parameter Interpolation:** The functions `compute_loss_surface_1d` and `compute_loss_surface_2d` temporarily modify the model's parameters to explore the loss surface. The original parameters are restored after computation.
2.  **Random Directions:** `create_random_direction` generates a random direction vector. For more advanced analysis, users might want to use specific directions (e.g., principal components of the Hessian, or directions defined by optimization trajectories).
3.  **Visualization:** This module provides the computation of loss values. External libraries like `matplotlib` are required for actual plotting and visualization, as demonstrated in the example.
