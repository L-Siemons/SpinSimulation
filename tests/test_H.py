import spinSimulations.spin as ss
from spinSimulations.solver_H import prop_h, com_h, traj_prop_h

import numpy as np
import itertools

results = {
    
}

class TestRotation:
    def test_simple_rot_dense(self):
        is_sparse = False        

        s = ss.System(1, is_sparse=is_sparse)
        op = lambda label : s.op(0, label)
        
        # Several specific test to check the sign
        assert np.allclose(
            -op("y"),
            prop_h(op("z"), op("x"), np.pi / 2)
        )
        
        assert np.allclose(
            op("x"),
            prop_h(op("z"), op("y"), np.pi / 2)
        )
        
        # General test for commutation
        for l1, l2 in itertools.permutations(['x', 'y', 'z'], 2):
            assert np.allclose(
                1j * com_h(op(l1), op(l2)),
                prop_h(op(l1), op(l2), np.pi / 2)
            )
                
    def test_simple_rot_sparse(self):
        is_sparse = True
        s = ss.System(1, is_sparse=is_sparse)
        op = lambda label : s.op(0, label)
        
        # Several specific test to check the sign
        assert np.allclose(
            -op("y").toarray(),
            prop_h(op("z"), op("x"), np.pi / 2).toarray()
        )
        
        assert np.allclose(
            op("x").toarray(),
            prop_h(op("z"), op("y"), np.pi / 2).toarray()
        )
        
        # General test for commutation
        for l1, l2 in itertools.permutations(['x', 'y', 'z'], 2):
            assert np.allclose(
                1j * com_h(op(l1), op(l2)).toarray(),
                prop_h(op(l1), op(l2), np.pi / 2).toarray()
            )
            
    def test_nutation(self):
        is_sparse = False
        s = ss.System(1, is_sparse=is_sparse)
        op = lambda label : s.op(0, label)

        nu = 20
        times = np.linspace(0, 5 / nu, 200)
        
        traj_z, traj_x = traj_prop_h(
            op("z"), 2 * np.pi * nu * op("y"), times, [op("z"), op("x")], s
        )
        
        assert np.allclose(
            traj_z,
            np.cos(2 * np.pi * nu * times)
        )
        
        assert np.allclose(
            traj_x,
            np.sin(2 * np.pi * nu * times)
        )
        
        
    
    # def test_slic