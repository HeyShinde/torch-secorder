# torch-secorder

A PyTorch-native library for efficient second-order computations in deep neural networks.

## Overview

Torch-Secorder provides efficient implementations of second-order optimization utilities for PyTorch, including:

- Hessian-Vector Products (HVP)
- Jacobian-Vector Products (JVP)
- Vector-Jacobian Products (VJP)
- Gauss-Newton matrix computations
- Hessian trace estimation

These tools are essential for:
- Second-order optimization methods
- Natural gradient descent
- Curvature-based regularization
- Neural network analysis and debugging

## Features

- **Hessian-Vector Products (HVP)**
  - Computation of Hv for any vector v
  - Trace estimation using Hutchinson's method
  - Support for distributed computation
  - Automatic parameter handling

- **Hessian Diagonal**
  - Computation of Hessian diagonal elements
  - Trace estimation using diagonal elements
  - Support for custom vectors
  - Gradient requirement validation

- **Trace Estimation**
  - HVP-based: Uses Hutchinson's method with random vectors
  - Diagonal-based: Uses exact diagonal elements
  - Both methods compute the same quantity
  - Choose based on model size and accuracy requirements

- **Model Integration**
  - PyTorch model integration
  - Parameter management
  - Loss function support
  - Model-specific computations

## Installation

```bash
pip install torch-secorder
```

Or install from source:
```bash
git clone https://github.com/pybrainn/torch-secorder.git
cd torch-secorder
pip install -e .
```

## Quick Start

See our [documentation](https://torch-secorder.readthedocs.io/) for detailed examples and tutorials.

## Documentation

- [API Reference](https://torch-secorder.readthedocs.io/en/latest/api/core.html)
- [Examples](https://torch-secorder.readthedocs.io/en/latest/examples.html)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{torch_secorder2025,
  author = PyBrainn,
  title = {Torch-Secorder: Second-Order Optimization for PyTorch},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/pybrainn/torch-secorder}
}
```
