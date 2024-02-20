import spinSimulations.spin as s
import numpy as np

from single_spins_ops import single_spin_ops

class TestSingleSpinDense:
    
    labels = ['x', 'y', 'z', 'p', 'm']
    is_sparse = False
    
    def test_half(self):
        sys = s.System(1, is_sparse=self.is_sparse)
        
        for label in self.labels:
            assert np.allclose(
                sys.op_single(1/2, label),
                single_spin_ops['1/2'][label].astype(sys.dtype)
            )
            
    def test_one(self):
        sys = s.System(1, is_sparse=self.is_sparse)
        
        for label in self.labels:
            assert np.allclose(
                sys.op_single(1, label),
                single_spin_ops['1'][label].astype(sys.dtype)
            )
            
    def test_one_and_half(self):
        sys = s.System(1, is_sparse=self.is_sparse)
        
        for label in self.labels:
            assert np.allclose(
                sys.op_single(3/2, label),
                single_spin_ops['3/2'][label].astype(sys.dtype)
            )
            
    def test_two(self):
        sys = s.System(1, is_sparse=self.is_sparse)
        
        for label in self.labels:
            assert np.allclose(
                sys.op_single(2, label),
                single_spin_ops['2'][label].astype(sys.dtype)
            )

class TestSingleSpinSparse:
    
    labels = ['x', 'y', 'z', 'p', 'm']
    is_sparse = True
    
    def test_half(self):
        sys = s.System(1, is_sparse=self.is_sparse)
        
        for label in self.labels:
            assert np.allclose(
                sys.op_single(1/2, label).toarray(),
                single_spin_ops['1/2'][label].astype(sys.dtype)
            )
            
    def test_one(self):
        sys = s.System(1, is_sparse=self.is_sparse)
        
        for label in self.labels:
            assert np.allclose(
                sys.op_single(1, label).toarray(),
                single_spin_ops['1'][label].astype(sys.dtype)
            )
            
    def test_one_and_half(self):
        sys = s.System(1, is_sparse=self.is_sparse)
        
        for label in self.labels:
            assert np.allclose(
                sys.op_single(3/2, label).toarray(),
                single_spin_ops['3/2'][label].astype(sys.dtype)
            )
            
    def test_two(self):
        sys = s.System(1, is_sparse=self.is_sparse)
        
        for label in self.labels:
            assert np.allclose(
                sys.op_single(2, label).toarray(),
                single_spin_ops['2'][label].astype(sys.dtype)
            )