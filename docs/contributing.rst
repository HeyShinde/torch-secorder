Contributing
============

We welcome contributions to torch-secorder! This guide will help you get started.

Development Setup
-----------------

1. Fork the repository
2. Clone your fork:

.. code-block:: bash
    :caption: Clone the forked repository from your profile

    git clone https://github.com/yourusername/torch-secorder.git
    cd torch-secorder

3. Install development dependencies:

.. code-block:: bash
    :caption: Install dependencies required for development and document

    poetry install --with dev,docs

4. Make the changes: Fix the bugs 🐛, add features 🚀...

5. Run the checks before submitting:

.. code-block:: bash
    :caption: Run pre-commit before Committing

        poetry run pre-commit clean
        poetry run pre-commit install --install-hooks
        poetry run pre-commit run --all-files

6. Create a new branch:

.. code-block:: bash
    :caption: Create a branch with suitable name

    git checkout -b feature/your-feature-name

7. Push code:

.. code-block:: bash
    :caption: push to the new branch

    git push origin feature/your-feature-name

8. Request a PR:
   Provide a detailed comment for your Pull Request

Code Style
----------

We use:
- Ruff for code linting & formatting
- isort for import sorting
- mypy for type checking
- pytest for testing

Documentation
-------------

1. Add docstrings to new functions/classes
2. Update relevant documentation files
3. Build and check the docs:

.. code-block:: bash
    :caption: Build documentation files

        cd docs
        poetry run make html

Pull Request Process
--------------------

1. Update the README.md with details of changes if needed
2. Update the documentation
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

Bug Reports
-----------

When filing a bug report, please include:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

Code of Conduct
---------------

Please be respectful and considerate of others when contributing. We have adopted the
`Contributor Covenant <https://www.contributor-covenant.org/version/2/0/code_of_conduct.html>`_
as our Code of Conduct. Please read the full text in our `CODE_OF_CONDUCT.md <https://github.com/pybrainn/torch-secorder/blob/main/CODE_OF_CONDUCT.md>`_
file.

For more detailed information about contributing, including our full Code of Conduct,
please see our `CONTRIBUTING.md <https://github.com/pybrainn/torch-secorder/blob/main/CONTRIBUTING.md>`_
file on GitHub.
