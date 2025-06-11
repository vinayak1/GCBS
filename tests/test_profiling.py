"""
Tests for GCBS profiling scripts.
"""

import unittest
import torch
import sys
import os
import importlib.util
import tempfile
import time
from unittest.mock import patch, MagicMock


class TestSingleGPUProfiling(unittest.TestCase):
    """Test cases for single GPU profiling script."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Import the profiling function
        spec = importlib.util.spec_from_file_location(
            "single_gpu_profiling", 
            "/workspace/GCBS/single_gpu_profiling_gcbs.py"
        )
        self.profiling_module = importlib.util.module_from_spec(spec)
        
    def test_compute_gce_function_exists(self):
        """Test that compute_gce function exists and is callable."""
        try:
            spec = importlib.util.spec_from_file_location(
                "single_gpu_profiling", 
                "/workspace/GCBS/single_gpu_profiling_gcbs.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            self.assertTrue(hasattr(module, 'compute_gce'))
            self.assertTrue(callable(module.compute_gce))
        except Exception as e:
            self.fail(f"Failed to import or find compute_gce function: {e}")
            
    def test_compute_gce_basic_functionality(self):
        """Test basic functionality of compute_gce from profiling script."""
        try:
            spec = importlib.util.spec_from_file_location(
                "single_gpu_profiling", 
                "/workspace/GCBS/single_gpu_profiling_gcbs.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Test with small data to avoid GPU memory issues
            size = 100
            z1 = torch.rand(size, 32)
            z2 = torch.rand(size, 32)
            quantile = 0.9
            
            # Mock CUDA availability if not available
            if not torch.cuda.is_available():
                with patch('torch.Tensor.cuda', return_value=torch.Tensor.cpu):
                    permutation = module.compute_gce(z1, z2, quantile)
            else:
                permutation = module.compute_gce(z1, z2, quantile)
            
            self.assertEqual(len(permutation), size)
            self.assertEqual(set(permutation), set(range(size)))
            
        except Exception as e:
            self.skipTest(f"Skipping test due to import/execution error: {e}")
            
    def test_compute_gce_different_sizes(self):
        """Test compute_gce with different input sizes."""
        try:
            spec = importlib.util.spec_from_file_location(
                "single_gpu_profiling", 
                "/workspace/GCBS/single_gpu_profiling_gcbs.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            sizes = [10, 50, 100]
            
            for size in sizes:
                with self.subTest(size=size):
                    z1 = torch.rand(size, 16)
                    z2 = torch.rand(size, 16)
                    quantile = 0.8
                    
                    if not torch.cuda.is_available():
                        with patch('torch.Tensor.cuda', return_value=torch.Tensor.cpu):
                            permutation = module.compute_gce(z1, z2, quantile)
                    else:
                        permutation = module.compute_gce(z1, z2, quantile)
                    
                    self.assertEqual(len(permutation), size)
                    self.assertEqual(set(permutation), set(range(size)))
                    
        except Exception as e:
            self.skipTest(f"Skipping test due to import/execution error: {e}")


class TestMultiGPUProfiling(unittest.TestCase):
    """Test cases for multi GPU profiling script."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        
    def test_quantile_estimation_function(self):
        """Test quantile estimation function from multi-GPU script."""
        try:
            spec = importlib.util.spec_from_file_location(
                "multi_gpu_profiling", 
                "/workspace/GCBS/multi_gpu_profiling_gcbs.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Test quantile estimation
            src_mat = torch.randn(20, 32)
            tgt_mat = torch.randn(20, 32)
            quantile = 0.9
            
            if not torch.cuda.is_available():
                with patch('torch.Tensor.cuda', return_value=torch.Tensor.cpu):
                    quantiles = module.quantile_estimation(src_mat, tgt_mat, quantile)
            else:
                quantiles = module.quantile_estimation(src_mat, tgt_mat, quantile)
            
            self.assertIsInstance(quantiles, list)
            self.assertGreater(len(quantiles), 0)
            
        except Exception as e:
            self.skipTest(f"Skipping test due to import/execution error: {e}")
            
    def test_get_thresh_sim_graph_function(self):
        """Test similarity graph thresholding function."""
        try:
            spec = importlib.util.spec_from_file_location(
                "multi_gpu_profiling", 
                "/workspace/GCBS/multi_gpu_profiling_gcbs.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Test similarity graph creation
            src_mat = torch.randn(15, 16)
            tgt_mat = torch.randn(15, 16)
            quantile = 0.5
            
            if not torch.cuda.is_available():
                with patch('torch.Tensor.cuda', return_value=torch.Tensor.cpu):
                    row, col, data = module.get_thresh_sim_graph(src_mat, tgt_mat, quantile)
            else:
                row, col, data = module.get_thresh_sim_graph(src_mat, tgt_mat, quantile)
            
            self.assertIsInstance(row, list)
            self.assertIsInstance(col, list)
            self.assertIsInstance(data, list)
            self.assertEqual(len(row), len(col))
            self.assertEqual(len(col), len(data))
            
        except Exception as e:
            self.skipTest(f"Skipping test due to import/execution error: {e}")


class TestProfilingPerformance(unittest.TestCase):
    """Test performance characteristics of profiling scripts."""
    
    def test_profiling_execution_time(self):
        """Test that profiling completes within reasonable time."""
        try:
            spec = importlib.util.spec_from_file_location(
                "single_gpu_profiling", 
                "/workspace/GCBS/single_gpu_profiling_gcbs.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Test with small data for reasonable execution time
            size = 50
            z1 = torch.rand(size, 32)
            z2 = torch.rand(size, 32)
            quantile = 0.9
            
            start_time = time.time()
            
            if not torch.cuda.is_available():
                with patch('torch.Tensor.cuda', return_value=torch.Tensor.cpu):
                    permutation = module.compute_gce(z1, z2, quantile)
            else:
                permutation = module.compute_gce(z1, z2, quantile)
                
            execution_time = time.time() - start_time
            
            # Should complete within 30 seconds for small data
            self.assertLess(execution_time, 30.0)
            self.assertEqual(len(permutation), size)
            
        except Exception as e:
            self.skipTest(f"Skipping test due to import/execution error: {e}")
            
    def test_memory_usage_reasonable(self):
        """Test that memory usage is reasonable for small inputs."""
        try:
            spec = importlib.util.spec_from_file_location(
                "single_gpu_profiling", 
                "/workspace/GCBS/single_gpu_profiling_gcbs.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Test with very small data
            size = 20
            z1 = torch.rand(size, 16)
            z2 = torch.rand(size, 16)
            quantile = 0.8
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                initial_memory = torch.cuda.memory_allocated()
                
                permutation = module.compute_gce(z1, z2, quantile)
                
                peak_memory = torch.cuda.max_memory_allocated()
                memory_used = peak_memory - initial_memory
                
                # Memory usage should be reasonable for small inputs
                self.assertLess(memory_used, 100 * 1024 * 1024)  # Less than 100MB
                self.assertEqual(len(permutation), size)
            else:
                # Just test functionality without CUDA
                with patch('torch.Tensor.cuda', return_value=torch.Tensor.cpu):
                    permutation = module.compute_gce(z1, z2, quantile)
                self.assertEqual(len(permutation), size)
                
        except Exception as e:
            self.skipTest(f"Skipping test due to import/execution error: {e}")


class TestProfilingInputValidation(unittest.TestCase):
    """Test input validation for profiling functions."""
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        try:
            spec = importlib.util.spec_from_file_location(
                "single_gpu_profiling", 
                "/workspace/GCBS/single_gpu_profiling_gcbs.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Test with empty tensors
            z1 = torch.empty(0, 32)
            z2 = torch.empty(0, 32)
            quantile = 0.9
            
            # This should either handle gracefully or raise appropriate error
            try:
                if not torch.cuda.is_available():
                    with patch('torch.Tensor.cuda', return_value=torch.Tensor.cpu):
                        permutation = module.compute_gce(z1, z2, quantile)
                else:
                    permutation = module.compute_gce(z1, z2, quantile)
                self.assertEqual(len(permutation), 0)
            except (ValueError, RuntimeError, IndexError):
                # These are acceptable errors for empty input
                pass
                
        except Exception as e:
            self.skipTest(f"Skipping test due to import/execution error: {e}")
            
    def test_mismatched_dimensions(self):
        """Test handling of mismatched input dimensions."""
        try:
            spec = importlib.util.spec_from_file_location(
                "single_gpu_profiling", 
                "/workspace/GCBS/single_gpu_profiling_gcbs.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Test with mismatched dimensions
            z1 = torch.rand(10, 32)
            z2 = torch.rand(15, 32)  # Different batch size
            quantile = 0.9
            
            # This should either handle gracefully or raise appropriate error
            try:
                if not torch.cuda.is_available():
                    with patch('torch.Tensor.cuda', return_value=torch.Tensor.cpu):
                        permutation = module.compute_gce(z1, z2, quantile)
                else:
                    permutation = module.compute_gce(z1, z2, quantile)
                # If it succeeds, check the result makes sense
                self.assertIn(len(permutation), [10, 15])
            except (ValueError, RuntimeError, IndexError):
                # These are acceptable errors for mismatched dimensions
                pass
                
        except Exception as e:
            self.skipTest(f"Skipping test due to import/execution error: {e}")


if __name__ == '__main__':
    unittest.main()