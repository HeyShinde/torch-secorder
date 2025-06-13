Installation
============

Requirements
------------

- Python 3.11 or higher
- PyTorch 2.7 or higher
- NumPy

Version Compatibility
---------------------

torch-secorder is currently in version 0.0.1 (pre-release). The following version compatibility matrix is maintained:

+---------------+----------------+----------------+
| torch-secorder| PyTorch        | Python         |
+===============+================+================+
| 0.0.1         | >=2.7.1, <3.0  | >=3.11, <3.12  |
+---------------+----------------+----------------+

Note: This is a pre-release version. The API may change before the first stable release.

Installing from PyPI
--------------------

The easiest way to install torch-secorder is using pip:

.. code-block:: bash

    pip install torch-secorder

Installing from Source
----------------------

To install from source, clone the repository and install in development mode:

.. code-block:: bash

    git clone https://github.com/pybrainn/torch-secorder.git
    cd torch-secorder
    pip install -e .

Verifying Installation
----------------------

To verify the installation, you can run the test suite:

.. code-block:: bash

    pytest tests/

Or try a simple example:

.. code-block:: python

    import torch
    import torch.nn as nn
    from torch_secorder.core import hvp

    # Create a simple model
    model = nn.Linear(10, 1)
    x = torch.randn(1, 10)
    y = torch.randn(1, 1)

    # Compute HVP
    v = [torch.randn_like(p) for p in model.parameters()]
    hvp_result = hvp.model_hvp(model, nn.MSELoss(), x, y, v)
    print(hvp_result)
