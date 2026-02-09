# Contributing to lexprep

Thanks for your interest in contributing to lexprep.  
This document outlines the basic workflow and expectations for contributions.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone 
   cd lexprep
   ```
3. Create a virtual environment and install development dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -e ".[dev]"
   ```

## Development Workflow

### Making Changes

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following the code style guidelines below

3. Add or update tests as needed:
   ```bash
   pytest tests/
   ```

4. Ensure all tests pass before opening a pull request:
```bash
pytest 
```

5. Commit your changes with a clear, descriptive commit message:
   ```bash
   git commit -m "Add feature: description of your changes"
   ```

6. Push to your fork and submit a pull request

## Code Style Guidelines

- Follow PEP 8 style guidelines
- Use type hints for function arguments and return values
- Add docstrings for public functions and CLI-facing code
- Keep functions focused and modular
- Use descriptive variable names


## Adding New Features

When adding new features:

1. **Discuss First**: For major features, open an issue first to discuss the approach
2. **Update Documentation**: Update README.md and, where relevant, the CLI help text or output documentation
3. **Add Examples**: Include usage examples in `examples/` if applicable
4. **Add Tests**: Ensure comprehensive test coverage
5. **Update CLI**: If adding CLI commands, update help text appropriately

## CLI Command Guidelines

When adding new CLI commands:

- Use clear, descriptive command names
- Provide helpful `--help` text for all options
- Use consistent naming patterns (e.g., `--word-col`, `--output-path`)
- Support multiple file formats where appropriate (txt, csv, xlsx)
- Handle errors gracefully with informative messages

## Bug Reports

When reporting bugs, include:

- Python version and OS
- Full error traceback
- Minimal reproducible example
- Expected vs. actual behavior


## Questions?

Feel free to open an issue for any questions or concerns about contributing.

## License

By contributing to lexprep, you agree that your contributions will be licensed under the MIT License.
