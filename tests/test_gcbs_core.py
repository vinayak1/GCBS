"""
Tests for the core GCBS (Global Contrastive Batch Sampling) algorithm.
"""

import unittest
import torch
import numpy as np
import math
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse import csr_matrix
import statistics


def compute_gcbs(z1, z2, quantile):
    """
    Core GCBS algorithm implementation for testing.
    Based on the implementation from the README.
    """
    # (1) Stack and normalize outputs
    src_train_full = torch.nn.functional.normalize(z1).cuda() if torch.cuda.is_available() else torch.nn.functional.normalize(z1)
    tgt_train_full = torch.nn.functional.normalize(z2).cuda() if torch.cuda.is_available() else torch.nn.functional.normalize(z2)
    z1, z2 = [], []

    # (2) Estimate quantile
    chunk_size, num_samples, quantiles = 10, len(tgt_train_full), []
    for chunk_idx in range(math.ceil(len(tgt_train_full)/chunk_size)):
        mat_val = src_train_full[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size] @ tgt_train_full.T
        quantiles.append(float(torch.quantile(mat_val, quantile)))

    # (3) Get similarity graph thresholded on quantile
    row, col, data, quantile = [], [], [], statistics.median(quantiles)
    for chunk_idx in range(math.ceil(len(src_train_full)/chunk_size)):
        mat_val = src_train_full[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size] @ tgt_train_full.T
        ret = (mat_val.flatten() > quantile).nonzero(as_tuple=True)[0].cpu()
        row += ((ret - (ret % num_samples))/num_samples + chunk_idx*chunk_size).int().tolist()
        col += (ret % num_samples).tolist()
        data += [1.0 for _ in range(len(ret))]

    # (4) Get permutation using graph bandwidth minimization on sparsified graph (cuthill-mckee)
    return list(reverse_cuthill_mckee(csr_matrix((data, (row, col)),
                                                  shape=(num_samples, num_samples))))


class TestGCBSCore(unittest.TestCase):
    """Test cases for the core GCBS algorithm."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
        
    def test_compute_gcbs_basic_functionality(self):
        """Test basic functionality of compute_gcbs."""
        # Create simple test data
        batch_size = 100
        embedding_dim = 64
        z1 = torch.randn(batch_size, embedding_dim)
        z2 = torch.randn(batch_size, embedding_dim)
        quantile = 0.9
        
        # Run GCBS
        permutation = compute_gcbs(z1, z2, quantile)
        
        # Check output properties
        self.assertEqual(len(permutation), batch_size)
        self.assertEqual(set(permutation), set(range(batch_size)))
        self.assertIsInstance(permutation, list)
        
    def test_compute_gcbs_different_quantiles(self):
        """Test GCBS with different quantile values."""
        batch_size = 50
        embedding_dim = 32
        z1 = torch.randn(batch_size, embedding_dim)
        z2 = torch.randn(batch_size, embedding_dim)
        
        quantiles = [0.5, 0.8, 0.9, 0.95, 0.99]
        
        for quantile in quantiles:
            with self.subTest(quantile=quantile):
                permutation = compute_gcbs(z1, z2, quantile)
                self.assertEqual(len(permutation), batch_size)
                self.assertEqual(set(permutation), set(range(batch_size)))
                
    def test_compute_gcbs_identical_embeddings(self):
        """Test GCBS with identical embeddings."""
        batch_size = 30
        embedding_dim = 16
        z1 = torch.ones(batch_size, embedding_dim)
        z2 = torch.ones(batch_size, embedding_dim)
        quantile = 0.9
        
        permutation = compute_gcbs(z1, z2, quantile)
        
        self.assertEqual(len(permutation), batch_size)
        self.assertEqual(set(permutation), set(range(batch_size)))
        
    def test_compute_gcbs_orthogonal_embeddings(self):
        """Test GCBS with orthogonal embeddings."""
        batch_size = 32
        embedding_dim = 32
        
        # Create orthogonal embeddings
        z1 = torch.eye(batch_size, embedding_dim)
        z2 = torch.eye(batch_size, embedding_dim)
        quantile = 0.9
        
        permutation = compute_gcbs(z1, z2, quantile)
        
        self.assertEqual(len(permutation), batch_size)
        self.assertEqual(set(permutation), set(range(batch_size)))
        
    def test_compute_gcbs_small_batch(self):
        """Test GCBS with very small batch size."""
        batch_size = 5
        embedding_dim = 8
        z1 = torch.randn(batch_size, embedding_dim)
        z2 = torch.randn(batch_size, embedding_dim)
        quantile = 0.8
        
        permutation = compute_gcbs(z1, z2, quantile)
        
        self.assertEqual(len(permutation), batch_size)
        self.assertEqual(set(permutation), set(range(batch_size)))
        
    def test_compute_gcbs_large_batch(self):
        """Test GCBS with larger batch size."""
        batch_size = 500
        embedding_dim = 128
        z1 = torch.randn(batch_size, embedding_dim)
        z2 = torch.randn(batch_size, embedding_dim)
        quantile = 0.95
        
        permutation = compute_gcbs(z1, z2, quantile)
        
        self.assertEqual(len(permutation), batch_size)
        self.assertEqual(set(permutation), set(range(batch_size)))
        
    def test_compute_gcbs_deterministic(self):
        """Test that GCBS is deterministic given same inputs."""
        batch_size = 50
        embedding_dim = 64
        
        torch.manual_seed(123)
        z1 = torch.randn(batch_size, embedding_dim)
        z2 = torch.randn(batch_size, embedding_dim)
        quantile = 0.9
        
        permutation1 = compute_gcbs(z1.clone(), z2.clone(), quantile)
        permutation2 = compute_gcbs(z1.clone(), z2.clone(), quantile)
        
        self.assertEqual(permutation1, permutation2)
        
    def test_compute_gcbs_normalized_inputs(self):
        """Test that GCBS works correctly with pre-normalized inputs."""
        batch_size = 40
        embedding_dim = 32
        
        # Create normalized embeddings
        z1 = torch.randn(batch_size, embedding_dim)
        z2 = torch.randn(batch_size, embedding_dim)
        z1_norm = torch.nn.functional.normalize(z1)
        z2_norm = torch.nn.functional.normalize(z2)
        quantile = 0.9
        
        permutation1 = compute_gcbs(z1, z2, quantile)
        permutation2 = compute_gcbs(z1_norm, z2_norm, quantile)
        
        # Both should produce valid permutations
        self.assertEqual(len(permutation1), batch_size)
        self.assertEqual(len(permutation2), batch_size)
        self.assertEqual(set(permutation1), set(range(batch_size)))
        self.assertEqual(set(permutation2), set(range(batch_size)))
        
    def test_compute_gcbs_edge_quantiles(self):
        """Test GCBS with edge case quantile values."""
        batch_size = 20
        embedding_dim = 16
        z1 = torch.randn(batch_size, embedding_dim)
        z2 = torch.randn(batch_size, embedding_dim)
        
        # Test very low quantile
        permutation_low = compute_gcbs(z1, z2, 0.1)
        self.assertEqual(len(permutation_low), batch_size)
        self.assertEqual(set(permutation_low), set(range(batch_size)))
        
        # Test very high quantile
        permutation_high = compute_gcbs(z1, z2, 0.999)
        self.assertEqual(len(permutation_high), batch_size)
        self.assertEqual(set(permutation_high), set(range(batch_size)))
        
    def test_compute_gcbs_different_embedding_dims(self):
        """Test GCBS with different embedding dimensions."""
        batch_size = 30
        embedding_dims = [8, 16, 32, 64, 128, 256]
        quantile = 0.9
        
        for dim in embedding_dims:
            with self.subTest(embedding_dim=dim):
                z1 = torch.randn(batch_size, dim)
                z2 = torch.randn(batch_size, dim)
                
                permutation = compute_gcbs(z1, z2, quantile)
                
                self.assertEqual(len(permutation), batch_size)
                self.assertEqual(set(permutation), set(range(batch_size)))


class TestGCBSHelperFunctions(unittest.TestCase):
    """Test helper functions used in GCBS."""
    
    def test_normalization(self):
        """Test that normalization works correctly."""
        batch_size = 10
        embedding_dim = 8
        z = torch.randn(batch_size, embedding_dim)
        
        z_norm = torch.nn.functional.normalize(z)
        
        # Check that all vectors have unit norm
        norms = torch.norm(z_norm, dim=1)
        torch.testing.assert_close(norms, torch.ones(batch_size), atol=1e-6, rtol=1e-6)
        
    def test_quantile_computation(self):
        """Test quantile computation."""
        # Create a simple matrix
        matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        
        # Test different quantiles
        q50 = torch.quantile(matrix, 0.5)
        q90 = torch.quantile(matrix, 0.9)
        
        self.assertGreater(q90, q50)
        self.assertGreaterEqual(q50, matrix.min())
        self.assertLessEqual(q90, matrix.max())
        
    def test_sparse_matrix_creation(self):
        """Test sparse matrix creation for graph representation."""
        # Simple test data
        row = [0, 1, 2, 0]
        col = [1, 2, 0, 2]
        data = [1.0, 1.0, 1.0, 1.0]
        shape = (3, 3)
        
        sparse_matrix = csr_matrix((data, (row, col)), shape=shape)
        
        self.assertEqual(sparse_matrix.shape, shape)
        self.assertEqual(sparse_matrix.nnz, len(data))
        
    def test_cuthill_mckee_ordering(self):
        """Test Cuthill-McKee ordering."""
        # Create a simple graph
        row = [0, 1, 2, 3, 0, 1, 2]
        col = [1, 2, 3, 0, 0, 1, 2]
        data = [1.0] * len(row)
        shape = (4, 4)
        
        sparse_matrix = csr_matrix((data, (row, col)), shape=shape)
        permutation = list(reverse_cuthill_mckee(sparse_matrix))
        
        self.assertEqual(len(permutation), shape[0])
        self.assertEqual(set(permutation), set(range(shape[0])))


if __name__ == '__main__':
    unittest.main()