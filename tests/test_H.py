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
            
    def test_nutation_dense(self):
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
        
    def test_nutation_sparse(self):
        is_sparse = True
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
        
    def test_slic(self):
        Js = np.array(
            [[0, 15.7, 6.6],
            [0, 0, 3.2],
            [0, 0, 0]]
        )

        nuclei_list = 2 * ["1H"] + ["13C"]
        
        n_spins = 3
        tau_slic = 1 / np.abs(Js[0,2] - Js[1, 2])
        times = np.linspace(0, tau_slic * 5, 1_000)
         
        # no sparse
        s = ss.System(n_spins, is_sparse=False, nuclei_list=nuclei_list)
        
        ham = 0
        for idx_1 in range(n_spins):
            for idx_2 in range(idx_1 + 1 , n_spins):
                secular = False if s.nuclei_list[idx_1] == s.nuclei_list[idx_2] else True
                ham += 2 * np.pi * Js[idx_1, idx_2] * s.scalar(idx_1, idx_2, secular=secular)
                
        rho = s.singlet(0, 1)

        pol_level = s.pol_level(2, "x")

        ham_slic = - 2 * np.pi * Js[0 , 1] * s.op(2, "x")
        
        traj = traj_prop_h(
            rho, ham + ham_slic, times, pol_level, s
        )
        
        np.allclose(
            traj,
            0.5 * (np.sin(2 * np.pi * times / (2 * tau_slic)) + 1)
        )
        
        # sparse
        s = ss.System(n_spins, is_sparse=True, nuclei_list=nuclei_list)
        
        ham = 0
        for idx_1 in range(n_spins):
            for idx_2 in range(idx_1 + 1 , n_spins):
                secular = False if s.nuclei_list[idx_1] == s.nuclei_list[idx_2] else True
                ham += 2 * np.pi * Js[idx_1, idx_2] * s.scalar(idx_1, idx_2, secular=secular)
                
        rho = s.singlet(0, 1)

        pol_level = s.pol_level(2, "x")

        ham_slic = - 2 * np.pi * Js[0 , 1] * s.op(2, "x")
        traj = traj_prop_h(
            rho, ham + ham_slic, times, pol_level, s
        )
        
        np.allclose(
            traj,
            0.5 * (np.sin(2 * np.pi * times / (2 * tau_slic)) + 1)
        )