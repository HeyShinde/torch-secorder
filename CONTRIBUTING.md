# Contributing to torch-secorder

We love your input! We want to make contributing to torch-secorder as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Pull Request Process

1. Update the README.md with details of changes to the interface, if applicable.
2. Update the documentation with details of any new environment variables, exposed ports, useful file locations and container parameters.
3. The PR will be merged once you have the sign-off of at least one other developer.

## Any contributions you make will be under the Apache 2.0 Software License

In short, when you submit code changes, your submissions are understood to be under the same [Apache 2.0 License](https://choosealicense.com/licenses/apache-2.0/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using GitHub's [issue tracker](https://github.com/pybrainn/torch-secorder/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/pybrainn/torch-secorder/issues/new); it's that easy!

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/torch-secorder.git
   cd torch-secorder
   ```
3. Install development dependencies:
   ```bash
   poetry install --with dev,docs
   ```
4. Make your changes
5. Run the checks before submitting:
   ```bash
   poetry run pre-commit clean
   poetry run pre-commit install --install-hooks
   poetry run pre-commit run --all-files
   ```
6. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
7. Push your changes:
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

We use:
- Ruff for code linting & formatting
- isort for import sorting
- mypy for type checking
- pytest for testing

## Documentation

1. Add docstrings to new functions/classes
2. Update relevant documentation files
3. Build and check the docs:
   ```bash
   cd docs
   poetry run make html
   ```

## Code of Conduct

We have adopted the [Contributor Covenant](https://www.contributor-covenant.org) as our Code of Conduct. Please read the full text in our [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) file.
