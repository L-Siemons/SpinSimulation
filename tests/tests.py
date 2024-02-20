import spinSimulations.spin as s
import numpy as np

class TestSingleSpin:
    def test_half_z(self):
        sys = s.System(1, is_sparse=False)
        
        assert np.allclose(
            sys.op_single(1/2, "z"),
            0.5 * np.array(
                [[1, 0],
                 [0, -1]],
                dtype=sys.dtype
            )
        )
        
    def test_half_p(self):
        sys = s.System(1, is_sparse=False)
        
        assert np.allclose(
            sys.op_single(1/2, "p"),
            np.array(
                [[0, 1],
                 [0, 0]],
                dtype=sys.dtype
            )
        )
        
    def test_half_m(self):
        sys = s.System(1, is_sparse=False)
        
        assert np.allclose(
            sys.op_single(1/2, "m"),
            np.array(
                [[0, 0],
                 [1, 0]],
                dtype=sys.dtype
            )
        )
        
    def test_half_x(self):
        sys = s.System(1, is_sparse=False)
        
        assert np.allclose(
            sys.op_single(1/2, "x"),
            0.5 * np.array(
                [[0, 1],
                 [1, 0]],
                dtype=sys.dtype
            )
        )
        
    def test_half_y(self):
        sys = s.System(1, is_sparse=False)
        
        assert np.allclose(
            sys.op_single(1/2, "y"),
            -0.5 * 1j * np.array(
                [[0, 1],
                 [-1, 0]],
                dtype=sys.dtype
            )
        )