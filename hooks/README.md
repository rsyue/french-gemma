# Development Environment Hooks & Guidelines

This directory contains requirements and guidelines for the development environment. Always follow these rules during development and CI tasks:

## Python Commands Execution
All python-related commands must be run within the virtual environment. Ensure you activate the virtual environment beforehand:
```bash
source .venv/bin/activate
```

Examples:
- Run a Python script:
  ```bash
  source .venv/bin/activate && python src/main.py
  ```
- Run tests using pytest:
  ```bash
  source .venv/bin/activate && pytest tests/
  ```

## Installing Dependencies
To install any new packages, use the `uv` tool within the active virtual environment:
```bash
source .venv/bin/activate && uv pip install <dependency_name>
```

## Continuous Integration (CI)
When running CI checks, always run both the linter and the test suite:
- **Linting**: Run Ruff to check style and syntax:
  ```bash
  source .venv/bin/activate && ruff check .
  ```
- **Testing**: Run Pytest to verify correctness:
  ```bash
  source .venv/bin/activate && pytest tests/
  ```
