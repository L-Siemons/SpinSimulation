import spinSimulations.spin as s
import numpy as np

# TODO: these are simple tests
# General test case should be written
class TestOpFull:
    
    def test_1(self):
        is_sparse = False
        sys = s.System(2, is_sparse=is_sparse)
        assert np.allclose(
            sys.op_full('0z1x'),
            sys.op(0, 'z') @ sys.op(1, 'x')
        )
        
    def test_2(self):
        is_sparse = False
        sys = s.System(2, is_sparse=is_sparse, nuclei_list=['1H', '2H'])
        assert np.allclose(
            sys.op_full('0z1x'),
            sys.op(0, 'z') @ sys.op(1, 'x')
        )
        
    def test_3(self):
        is_sparse = True
        sys = s.System(2, is_sparse=is_sparse)
        assert np.allclose(
            sys.op_full('0z1x').toarray(),
            (sys.op(0, 'z') @ sys.op(1, 'x')).toarray()
        )
        
    def test_4(self):
        is_sparse = True
        sys = s.System(2, is_sparse=is_sparse, nuclei_list=['1H', '2H'])
        assert np.allclose(
            sys.op_full('0z1x').toarray(),
            (sys.op(0, 'z') @ sys.op(1, 'x')).toarray()
        )
        
    def test_dims(self):
        ss = s.System(2, nuclei_list=["1H", "2H"])
        assert 6 == ss.get_spin_dim()
        
        ss = s.System(3, nuclei_list=["1H", "2H", "31P"])
        assert 12 == ss.get_spin_dim()
        
        ss = s.System(3, nuclei_list=["1H", "2H", "14N"])
        assert 18 == ss.get_spin_dim()
        