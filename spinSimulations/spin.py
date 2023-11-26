# this module 
import sys
sys.path.append("/home/rai/Desktop/SpinSimulation/")

import spinSimulations.cython_extensions as cext
import numpy as np
import scipy.sparse as sparse
from scipy.sparse import linalg
import functools
import pkg_resources
import re



class System():
       def __init__(self, nspins, atom_types):
        self.nspins = nspins
        self.atom_types = atom_types
        self.small_identity = np.identity(2, dtype=np.cdouble)
        self.single_spin_dictionary = self._initialize_single_spin_dictionary()
        self.operators = {}
        self.rho = None
        self.gammas = None
        self.carriers_hz = None
        self.carriers_omega = None
        self.lamour_freq = None
        self.lamour_freq_hz = None

    def _initialize_single_spin_dictionary(self):
        ix0 = np.array([[0, 1], [1, 0]], dtype=np.cdouble) * 0.5
        iy0 = np.array([[0, -1j], [1j, 0]], dtype=np.cdouble) * 0.5
        iz0 = np.array([[1, 0], [0, -1]], dtype=np.cdouble) * 0.5

        ip0 = ix0 + 1j * iy0
        im0 = ix0 - 1j * iy0
        i00 = iz0 * np.sqrt(2)

        return {'x': ix0, 'y': iy0, 'z': iz0, 'p': ip0, 'm': im0, 'o': i00}

    def load_gammas(self):
        gammas_file = pkg_resources.resource_filename('spinSimulations', 'dat/gammas.dat')

        self.gammas = {}
        with open(gammas_file) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2 and parts[0][0] != '#':
                    self.gammas[parts[0]] = float(parts[1])

    def set_lamour_freq(self, field):
        if self.gammas is None:
            self.load_gammas()

        # Ensure that the atom types are in gammas
        assertion_message = 'Please ensure that the atom types are as in:\n {self.gammas_file}'
        for atom_type in set(self.atom_types):
            assert atom_type in self.gammas, assertion_message

        # get the frequencies 
        self.lamour_freq = {i: field * self.gammas[i] for i in set(self.atom_types)}
        self.lamour_freq_hz = {i: freq / (np.pi * 2) for i, freq in self.lamour_freq.items()}

    
    def get_freq_shift(self, atom_id, ppm, absolute=False, freq='hz'):
        atom_type = self.atom_types[atom_id]
        lamour_val = self.lamour_freq_hz[atom_type] if freq == 'hz' else self.lamour_freq[atom_type]
        shift = lamour_val * ppm  
        return shift if not absolute else lamour_val + shift

    def get_shift_hz(self, atom_id, ppm, absolute=False):
        return self.get_freq_shift(atom_id, ppm, absolute=absolute, freq='hz')

    def get_shift_omega(self, atom_id, ppm, absolute=False):
        return self.get_freq_shift(atom_id, ppm, absolute=absolute, freq='omega')

    def kron_all(self, array_list):
        return cext.kron_all_v2(array_list)

    def operator(self,name):
        '''
        Get the operator, or generate it if we have not used it yet
        '''

        if name in self.operators:
            return self.operators[name]

        else:
            operator_str = name.split('.')
            operator_str = [re.findall(r'\D+|\d+', a) for a in operator_str]
            operator_str = np.array(operator_str, dtype=str).flatten()
            
            new_shape = (len(operator_str) // 2, 2) if len(operator_str) != 2 else (1, 2)
            operator_str = np.reshape(operator_str, new_shape)
            operator_list = [self.small_identity for _ in range(self.nspins)]

            for i in operator_str:
                counter = int(i[0])
                operator_list[counter] = self.single_spin_dictionary[i[1]]
                
            self.operators[name] = self.kron_all(operator_list)
            return self.operators[name]
        
    def print_rho(self,):
         print(np.array_str(self.rho, precision=2, suppress_small=True))
            
    def calc_single_spin_rotation(self, phase, angle):
        
        c, S = np.cos(angle/2), np.sin(angle/2)
        if phase == 'x':
            return np.array([[c, -1j*s],[-1j*s, c]], dtype=np.cdouble)
        elif phase == 'y':
            return np.array([[c, -s],[s, c]], dtype=np.cdouble)
        elif phase == 'z':
            return np.array([[c-1j*s, 0],[0, c+1j*s]], dtype=np.cdouble)
        else:
            print('Phase for rotation is not recognised')
            
    def calc_rotation_mat(self, phase, angle, active='all'):
        rotation = self.calc_single_spin_rotation(phase, angle)
        
        if active == 'all':
            rotation_list = [rotation for _ in range(self.nspins)]
        
        # apply the rotation to onle the active spins
        else:
            rotation_list = [self.small_identity for i in range(self.nspins)]
            for i in active:
                i = int(i)
                #i = i-1
                rotation_list[i] = rotation

            expand_matrix = self.kron_all(rotation_list)
        return self.kron_all(rotation_list)
    
    def apply_rotation(self, phase, angle, active='all'):
        rotation = self.calc_rotation_mat(phase, angle, active=active)
        rotation_inv = np.linalg.inv(rotation)
        #print([rotation, self.rho, rotation_inv])
        self.rho = functools.reduce(np.matmul, [rotation, self.rho, rotation_inv])
    
    def apply_hamiltonian(self, hamiltonian, time, rho):
        pre_operator = cext.matrix_exp(-1j * hamiltonian * time)
        post_operator = np.linalg.inv(pre_operator)
        #post_operator = scipy.linalg.expm(1j*hamiltonian*time)
        return functools.reduce(np.matmul, [pre_operator, rho, post_operator])
    
    def apply_hamiltonian_to_self(self, hamiltonian, time):
        self.rho = self.apply_hamiltonian(hamiltonian, time, self.rho)

    def get_trace(self,):
        return np.trace(self.rho)
    
    def scalar_coupling_hamil(self, spin1, spin2, coupling):

        # we don't need to consider strong coupling if we have different nuclei types
        # If you really wanted you could also add a check for the shift difference vs coupling too 
        if self.atom_types[spin1] == self.atom_types[spin2]:
            axis = ['x','y','z']
        else:
            axis = ['z']

        i_operators = [self.operator(f'{spin1}{i}') for i in axis]
        s_operators = [self.operator(f'{spin2}{i}') for i in axis]
        
        operator_products = [np.matmul(i,j) for i,j in zip(i_operators, s_operators)]        
        total = coupling*np.sum(operator_products, axis=0)*2*np.pi
        return total
    
    def calc_fid(self, hamiltonian, time_array, detection_operators='all'):
        
        # do some setup
        fid = []
        if detection_operators == 'all':
            ops = [self.operator(f'{i}p') for i in range(self.nspins)]
            total_detection_operator = np.sum(ops, axis=0) 
        
        else:
            total_detection_operator = np.sum([self.operator(i) for i in detection_operators], axis=0)

        # the domain we will calculate 
        dtime = time_array[1] - time_array[0]
        points = len(time_array)

        # set up the sparse matricies, could remove this part if we 
        # move to only using sparse matricies, this would tidy things up generally
        sparse_rho = sparse.csc_matrix(self.rho)
        sparse_hamil = sparse.csc_matrix(hamiltonian)
        sparse_detection_operator = sparse.csc_matrix(total_detection_operator)

        # calculate the propergator
        prop_part = sparse.linalg.expm(-1j*sparse_hamil*dtime)
        prop_part_inv = sparse.linalg.inv(prop_part)

        # main loop
        for i in range(points):
            # no evolution at time 0 
            if i != 0:
                sparse_rho = prop_part @ sparse_rho @ prop_part_inv

            # get fid point
            detetct = sparse_rho.multiply(sparse_detection_operator)
            fid_i = detetct.sum()
            fid.append(fid_i)

        return time_array, fid