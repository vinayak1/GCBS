# GCBS Testing Documentation

## Overview

This document provides comprehensive information about the test suite added to the GCBS (Global Contrastive Batch Sampling) repository.

## Test Suite Structure

The test suite consists of 5 main test modules with over 50 individual test cases:

### 1. Core Algorithm Tests (`tests/test_gcbs_core.py`)
- **14 test cases** covering the core GCBS algorithm
- Tests the `compute_gcbs` function with various scenarios:
  - Basic functionality validation
  - Different quantile values (0.5, 0.8, 0.9, 0.95, 0.99)
  - Various batch sizes (5 to 500 samples)
  - Different embedding dimensions (8 to 256)
  - Edge cases (identical embeddings, orthogonal embeddings)
  - Determinism verification
  - Input validation

### 2. Profiling Tests (`tests/test_profiling.py`)
- **9 test cases** for profiling script validation
- Tests both single-GPU and multi-GPU implementations
- Performance and memory usage validation
- Input validation and error handling
- Note: Some tests require GPU and may be skipped in CPU-only environments

### 3. PyTorch Lightning Tests (`tests/test_pytorch_lightning.py`)
- **8 test cases** for PyTorch Lightning integration
- Tests SimCLR and MoCo model implementations
- ContrastiveDataModule functionality
- Data loader creation and GCBS integration
- Model forward pass validation

### 4. SimCSE Model Tests (`tests/test_simcse_models.py`)
- **12 test cases** for SimCSE model components
- Tests MLPLayer, Similarity, and Pooler components
- BertForCL and RobertaForCL model validation
- Contrastive learning function testing
- Integration tests for complete model pipeline

### 5. UniXcoder Tests (`tests/test_unixcoder.py`)
- **12 test cases** for UniXcoder implementation
- Preprocessing script validation
- Data structure and feature extraction testing
- GCBS integration with code search tasks
- Error handling and edge case validation

## Key Testing Features

### Comprehensive Coverage
- **Algorithm Correctness**: Validates GCBS produces valid permutations
- **Edge Cases**: Tests empty inputs, mismatched dimensions, extreme quantiles
- **Performance**: Ensures reasonable execution time and memory usage
- **Integration**: Tests GCBS with different model architectures
- **Error Handling**: Validates graceful handling of invalid inputs
- **Determinism**: Ensures reproducible results with fixed random seeds

### Environment Compatibility
- **GPU/CPU Flexibility**: Tests automatically detect CUDA availability
- **Dependency Mocking**: External dependencies are mocked for isolation
- **Cross-platform**: Tests work on different operating systems
- **CI/CD Ready**: Designed for continuous integration pipelines

### Test Quality
- **Isolated Tests**: Each test is independent and doesn't affect others
- **Clear Documentation**: Every test has descriptive docstrings
- **Meaningful Assertions**: Tests verify specific expected behaviors
- **Proper Setup/Teardown**: Tests use proper fixtures and cleanup

## Running Tests

### Quick Start
```bash
# Install dependencies
pip install torch scipy

# Run all core tests (most reliable)
python tests/test_runner.py --test-type core

# Run all tests
python tests/test_runner.py --test-type all
```

### Individual Test Categories
```bash
python tests/test_runner.py --test-type core        # Core algorithm
python tests/test_runner.py --test-type profiling   # Profiling scripts
python tests/test_runner.py --test-type lightning   # PyTorch Lightning
python tests/test_runner.py --test-type simcse      # SimCSE models
python tests/test_runner.py --test-type unixcoder   # UniXcoder
```

### Using unittest directly
```bash
python -m unittest tests.test_gcbs_core -v
python -m unittest tests.test_gcbs_core.TestGCBSCore.test_compute_gcbs_basic_functionality
```

## Test Results Summary

### Core Algorithm Tests ✅
- **All 14 tests pass** consistently
- Tests the fundamental GCBS algorithm implementation
- Validates correctness across different scenarios
- No external dependencies required

### Integration Tests ⚠️
- Most tests pass with proper mocking
- Some tests may be skipped due to missing optional dependencies
- GPU-specific tests are skipped in CPU-only environments
- All core functionality is validated

## Dependencies

### Required (for core tests)
- `torch >= 1.11.0`
- `scipy >= 1.7.0`
- `numpy >= 1.21.0`

### Optional (for specific test modules)
- `transformers >= 4.0.0` (for SimCSE/UniXcoder tests)
- `pytorch-lightning >= 1.5.0` (for Lightning tests)
- `pytest` (for advanced test features)

## Continuous Integration

### GitHub Actions Workflow
- Automated testing on multiple Python versions (3.8-3.11)
- Tests run on Ubuntu with CPU-only environment
- Coverage reporting with codecov integration
- Separate test categories for better debugging

### Test Reliability
- Core tests have 100% pass rate
- Integration tests use extensive mocking for reliability
- Tests are designed to be deterministic and reproducible
- Proper error handling for missing dependencies

## Adding New Tests

When extending GCBS functionality:

1. **Add corresponding tests** in the appropriate test file
2. **Follow naming conventions**: `test_<functionality_name>`
3. **Include edge cases** and error conditions
4. **Mock external dependencies** to keep tests isolated
5. **Add descriptive docstrings** explaining test purpose
6. **Update documentation** if adding new test categories

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

## Benefits of the Test Suite

### For Developers
- **Confidence**: Ensures changes don't break existing functionality
- **Documentation**: Tests serve as executable documentation
- **Debugging**: Clear test failures help identify issues quickly
- **Refactoring**: Safe code improvements with test coverage

### For Users
- **Reliability**: Validated implementation reduces bugs
- **Compatibility**: Tests ensure code works across environments
- **Examples**: Tests show how to use GCBS correctly
- **Trust**: Comprehensive testing builds confidence in the library

### For Research
- **Reproducibility**: Tests ensure consistent algorithm behavior
- **Validation**: Confirms implementation matches paper description
- **Benchmarking**: Performance tests help optimize the algorithm
- **Extension**: Test framework supports adding new features

## Future Improvements

### Potential Enhancements
- **Performance benchmarks**: Add timing and memory usage tests
- **Property-based testing**: Use hypothesis for more comprehensive testing
- **Integration testing**: Add end-to-end workflow tests
- **Stress testing**: Test with very large datasets
- **Visualization**: Add tests for result visualization and analysis

### Test Coverage Goals
- Achieve >90% code coverage across all modules
- Add tests for all public APIs
- Include tests for all configuration options
- Test all error conditions and edge cases

## Conclusion

The GCBS test suite provides comprehensive validation of the algorithm implementation across multiple use cases and environments. With over 50 test cases covering core functionality, integration scenarios, and edge cases, the test suite ensures the reliability and correctness of the GCBS implementation.

The tests are designed to be:
- **Comprehensive**: Cover all major functionality
- **Reliable**: Work consistently across environments
- **Maintainable**: Easy to understand and extend
- **Practical**: Provide real-world usage examples

This testing infrastructure supports the continued development and adoption of GCBS by providing confidence in the implementation's correctness and reliability.