"""
Test runner for GCBS test suite.
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_all_tests():
    """Run all tests in the test suite."""
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_specific_test(test_module):
    """Run tests from a specific module."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_module)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_core_tests():
    """Run only core GCBS algorithm tests."""
    return run_specific_test('test_gcbs_core')

def run_profiling_tests():
    """Run only profiling tests."""
    return run_specific_test('test_profiling')

def run_pytorch_lightning_tests():
    """Run only PyTorch Lightning tests."""
    return run_specific_test('test_pytorch_lightning')

def run_simcse_tests():
    """Run only SimCSE model tests."""
    return run_specific_test('test_simcse_models')

def run_unixcoder_tests():
    """Run only UniXcoder tests."""
    return run_specific_test('test_unixcoder')

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run GCBS tests')
    parser.add_argument('--test-type', choices=['all', 'core', 'profiling', 'lightning', 'simcse', 'unixcoder'], 
                        default='all', help='Type of tests to run')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print(f"Running {args.test_type} tests...")
    print("=" * 50)
    
    if args.test_type == 'all':
        success = run_all_tests()
    elif args.test_type == 'core':
        success = run_core_tests()
    elif args.test_type == 'profiling':
        success = run_profiling_tests()
    elif args.test_type == 'lightning':
        success = run_pytorch_lightning_tests()
    elif args.test_type == 'simcse':
        success = run_simcse_tests()
    elif args.test_type == 'unixcoder':
        success = run_unixcoder_tests()
    
    print("=" * 50)
    if success:
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Some tests failed!")
        sys.exit(1)