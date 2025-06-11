# GCBS Test Suite

This directory contains comprehensive tests for the Global Contrastive Batch Sampling (GCBS) implementation.

## Test Structure

The test suite is organized into several modules:

### Core Tests (`test_gcbs_core.py`)
- Tests for the core GCBS algorithm implementation
- Validates the `compute_gcbs` function with various input scenarios
- Tests helper functions like normalization, quantile computation, and sparse matrix operations
- Includes edge cases and performance characteristics

### Profiling Tests (`test_profiling.py`)
- Tests for single GPU and multi-GPU profiling scripts
- Validates performance characteristics and memory usage
- Tests input validation and error handling
- Includes timing and resource usage tests

### PyTorch Lightning Tests (`test_pytorch_lightning.py`)
- Tests for the PyTorch Lightning implementation with CIFAR-10
- Validates SimCLR and MoCo model implementations
- Tests the ContrastiveDataModule and GCBS integration
- Includes data loader creation and model forward pass tests

### SimCSE Model Tests (`test_simcse_models.py`)
- Tests for SimCSE model components (MLPLayer, Similarity, Pooler)
- Validates BertForCL and RobertaForCL model implementations
- Tests contrastive learning functions and sentence embedding functionality
- Includes integration tests for the complete model pipeline

### UniXcoder Tests (`test_unixcoder.py`)
- Tests for UniXcoder preprocessing and training scripts
- Validates data processing and feature extraction
- Tests GCBS integration with code search tasks
- Includes error handling and data structure validation

## Running Tests

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements-test.txt
```

### Running All Tests

```bash
python tests/test_runner.py --test-type all
```

### Running Specific Test Categories

```bash
# Core GCBS algorithm tests
python tests/test_runner.py --test-type core

# Profiling tests
python tests/test_runner.py --test-type profiling

# PyTorch Lightning tests
python tests/test_runner.py --test-type lightning

# SimCSE model tests
python tests/test_runner.py --test-type simcse

# UniXcoder tests
python tests/test_runner.py --test-type unixcoder
```

### Running Individual Test Files

```bash
# Run a specific test file
python -m unittest tests.test_gcbs_core

# Run with verbose output
python -m unittest tests.test_gcbs_core -v

# Run a specific test class
python -m unittest tests.test_gcbs_core.TestGCBSCore

# Run a specific test method
python -m unittest tests.test_gcbs_core.TestGCBSCore.test_compute_gcbs_basic_functionality
```

## Test Coverage

The test suite covers:

1. **Algorithm Correctness**: Validates that GCBS produces valid permutations
2. **Edge Cases**: Tests with empty inputs, mismatched dimensions, extreme quantiles
3. **Performance**: Ensures reasonable execution time and memory usage
4. **Integration**: Tests GCBS integration with different model architectures
5. **Error Handling**: Validates graceful handling of invalid inputs
6. **Determinism**: Ensures reproducible results with fixed random seeds

## Test Environment

### GPU vs CPU Testing
- Tests automatically detect CUDA availability
- GPU-specific operations are mocked when CUDA is not available
- All tests should pass on both GPU and CPU environments

### Dependencies
- Some tests may be skipped if optional dependencies are not available
- Core algorithm tests have minimal dependencies and should always run
- Model-specific tests may require transformers, pytorch-lightning, etc.

### Mock Usage
- External dependencies are mocked to ensure tests are isolated
- File system operations are mocked in preprocessing tests
- Model components are mocked when testing integration logic

## Adding New Tests

When adding new functionality to GCBS:

1. **Create corresponding tests** in the appropriate test file
2. **Follow naming conventions**: `test_<functionality_name>`
3. **Include edge cases** and error conditions
4. **Mock external dependencies** to keep tests isolated
5. **Add docstrings** explaining what each test validates
6. **Update this README** if adding new test categories

### Test Template

```python
def test_new_functionality(self):
    """Test description of what this validates."""
    # Setup
    input_data = create_test_data()
    
    # Execute
    result = function_under_test(input_data)
    
    # Verify
    self.assertEqual(result.property, expected_value)
    self.assertIsInstance(result, ExpectedType)
```

## Continuous Integration

These tests are designed to be run in CI/CD pipelines:

- All tests use mocking to avoid external dependencies
- Tests are deterministic with fixed random seeds
- Resource usage is kept minimal for CI environments
- Tests provide clear failure messages for debugging

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed and PYTHONPATH is set correctly
2. **CUDA Errors**: Tests should automatically handle missing CUDA, but check GPU memory if running on GPU
3. **Timeout Issues**: Large batch sizes in tests may cause timeouts; reduce test data size if needed
4. **Mock Failures**: Some tests heavily use mocking; ensure mock objects have required attributes

### Debug Mode

Run tests with verbose output to see detailed information:

```bash
python tests/test_runner.py --test-type all --verbose
```

For even more detailed debugging, run individual tests with unittest's verbose mode:

```bash
python -m unittest tests.test_gcbs_core.TestGCBSCore.test_compute_gcbs_basic_functionality -v
```