"""Contains the main class System which hold information and 
opperations about the spin system
"""

import numpy as np
import scipy as sp
import scipy.sparse as sps
import functools
import pkg_resources

import spindata as sd


class System:
    """Summary

        Attributes
    ----------
    nuclei_list : list
        A list of atom types in the spin system. Each position is associated with an atom ID,
        ie the position in this list, which is used to identify each atom in the system. This list
        is also used to determine gyromagnetic ratio. All the available atom types are listed in
        spinSimulations/dat/gammas.dat
    carriers_hz : float
        The carrier at a given moment in hertz (hz)
    carriers_omega : float
        The carrier at a given moment in angular frequency (w)
    gammas : dict
        contains the gyromagnetic rations listed in spinSimulations/dat/gammas.dat
    gammas_file : str
        path to the gyromagnetic ratios file
    i00 : numpy.ndarray
        Description
    im0 : numpy.ndarray
        Lowering operator in the raising/lowering basis for a single spin
    ip0 : numpy.ndarray
        Raising operator in the raising/lowering basis for a single spin
    ix0 : numpy.ndarray
        Ix operator in the Cartesian basis for a single spin
    iy0 : numpy.ndarray
        Iy operator in the Cartesian basis for a single spin
    iz0 : numpy.ndarray
        Iz operator in the Cartesian basis for a single spin
    lamour_freq : dict
        This contains all the Lamour frequencies at a given magnetic field (T)
    lamour_freq_hz : dict
        This contains all the Lamour frequencies at a given magnetic field (T) in hz
    n_spins : int
        number of spins in the system
    operators : dict
        This contains all the operators for the spin system that have been calculated so far
    rho : numpy.ndarray
        The density matrix of the system
    single_spin_dictionary : dict
        contains the operators for the single spins
    small_identity : numpy.ndarray
        A 2x2 identity matrix

    """

    def __init__(self, n_spins, nuclei_list=None, dtype=np.cdouble, is_sparse=True):
        """
        Initial set up

        Parameters
        ----------
        n_spins : int
            number of spins in the system
        nuclei_list : list
            A list of atom types in the spin system. Each position is associated with an atom ID,
            i.e., the position in this list, which is used to identify each atom in the system. This list
            is also used to determine the gyromagnetic ratio. All the available atom types are listed in
            spinSimulations/dat/gammas.dat
        """
        self.n_spins = n_spins
        
        if nuclei_list is None:
            self.nuclei_list = ["1H"] * n_spins
        else:
            self.nuclei_list = nuclei_list
        
        self.small_identity = np.identity(2, dtype=np.cdouble)
        self.operators = {}
        self.rho = None
        self.gammas = None
        self.carriers_hz = None
        self.carriers_omega = None
        self.lamour_freq = None
        self.lamour_freq_hz = None
        self.dtype = dtype
        self.set_sparse(is_sparse)

    def set_sparse(self, is_sparse=True):
        self.is_sparse = is_sparse
        self._set_matrix_operations()
        
    def _set_matrix_operations(self):
        if self.is_sparse:
            self.kron = sps.kron
            self.matmul = sparse_dot
            self.expm = sps.linalg.expm
            self.eye = sps.eye
        else:
            self.kron = np.kron
            self.matmul = np.matmul
            self.expm = sp.linalg.expm
            self.eye = np.eye
            
    def get_n_spins(self):
        return self.n_spins

    def get_spin_dim(self):
        res = 1
        for nuclei in self.nuclei_list:
            res *= int(2 * sd.spin(nuclei) + 1)
        return res
    
    # TODO: this function should cash its result or it should put results in the dict
    def op_single(self, S, label='z'):
        if label == 'z':
            if self.is_sparse:
                return sps.diags(self._proj_values(S), dtype=self.dtype)
            else:
                return np.diag(self._proj_values(S))
                
        if label == 'p':
            if self.is_sparse:
                values = self._ladder_values(S)
                m = n = len(values) + 1
                return sps.diags(values, offsets=1, shape=(m, n), dtype=self.dtype)
            else:
                return np.diag(self._ladder_values(S), k=1)
        
        if label == 'm':
            if self.is_sparse:
                values = self._ladder_values(S)
                m = n = len(values) + 1
                return sps.diags(values, offsets=-1, shape=(m, n), dtype=self.dtype)
            else:
                return np.diag(self._ladder_values(S), k=-1)
            
        if label == 'x':
            return 0.5 * (self.op_single(S, 'p') + self.op_single(S, 'm'))
            
        if label == 'y':
            return -0.5 * 1j * (self.op_single(S, 'p') - self.op_single(S, 'm'))
        
    def _proj_values(self, S):
        """
        Return all projections for a spin S in descending order
        Helper function to build 

        Parameters
        ----------

        S : int or float
            Spin value

        Returns
        -------
        numpy.ndarray
            Projections for a spin S in descending order
        """
        n_proj = int(2*S + 1)
        return np.array([
            S - m 
            for m in range(n_proj)
        ], dtype=self.dtype)
    
    def _ladder_values(self, S):
        """
        Return all values on \pm 1 diagonal of a ladder operators
        Helper function to build the ladder operators

        Parameters
        ----------

        S : int or float
            Spin value

        Returns
        -------
        numpy.ndarray
            Return all values on \pm 1 diagonal of a ladder operators
        """
        n_proj = int(2*S + 1)
        return np.array([
            np.sqrt(S*(S+1) - (m - S) * (m - S + 1))
            for m in range(n_proj - 1)
        ], dtype=self.dtype)

    def op_full(self, name):
        """
        Get the operator if is present in System.operators. Otherwise it is generated and added to the
        dictionary. The operators follow the format N1aaN2bb where N1 and N2 are the atom inxdexes and
        xx and bb are the keys in System.single_spin_dictionary. For example the name for the I1zI2z operator
        would be '0z1z' and I1zI2x would be '0z1x'.

        Parameters
        ----------
        name : str
            The name of the operator

        Returns
        -------
        numpy.ndarray
            The operator we asked for
        """
        
        idx = list(map(int, name[::2]))
        idx_to_label = dict(
            zip(idx, name[1::2])
        )
        
        res = 1
        
        for i, nucleus in enumerate(self.nuclei_list):
            
            spin = sd.spin(nucleus)
            
            if i in idx:
                label = idx_to_label[i]
                res = self.kron(
                    res, 
                    self.op_single(spin, label=label)
                )
            else:
                res = self.kron(res, self.eye(int(2 * spin + 1)))
                    
        return res
        
        
    def op(self, idx, label="z"):
        """Return the operator for a spin system

        Parameters
        ----------
            idx : int 
                Spin index (! In python, count from zero)
            label :str 
                operator type, "x", "y", "z", "p", "m".

        Returns
        -------
        numpy.ndarray
            spin operator
        """
        
        # List of idx brunch
        if type(idx) == list:
            res = 0
            for id in idx:
                res += self._op(id, label=label)
            return res
        
        # idx as an int brunch
        return self._op(idx=idx, label=label)
    
    def _op(self, idx, label="z"):
        """Return the operator for a spin system

        Args:
            idx (int): Spin index (! In python, count from zero)
            label (str): operator type, "x", "y", "z", "p", "m".

        Returns:
            ndarray: spin operator
        """
        
        # handling incorrect input
        self._index_check(idx)
        
        res_op = 1  # this is a trick to write pretier code
        
        for i, nuclei in enumerate(self.nuclei_list):
                
            spin = sd.spin(nuclei)
            
            if i != idx:
                res_op = self.kron(res_op, self.eye(int(2*spin + 1)))
            else:
                res_op = self.kron(res_op, self.op_single(spin, label=label))    
        
        return res_op

    def _index_check(self, idx):
        if idx < 0:
            raise Exception("Index for a spin operator cannot be negative")
        
        if idx >= self.n_spins:
            raise Exception("Index for a spin operator is out of range")
        
    def scalar(self, i: int, j: int, secular=False):
        """
        Return the operator corresponding to scalar coupling betweem spin i and j

        Parameters
        ----------
        i : int
            Idx of the first spin
        j : int
            Idx of the second spins
        secular : bool, optional
            If true, only 'izjz' is returnd, otherwise 'ixjx' + 'iyjy' + 'izjz' 

        Returns
        -------
        numpy.ndarray
            The operator we asked for
        """
        self._index_check(i)
        self._index_check(j)
        
        if secular:
            return self.matmul(self.op(i, "z"), self.op(j, "z"))
    
        res = 0
        for coord in ('x', 'y', 'z'):
            res += self.matmul(self.op(i, coord), self.op(j, coord))
        return res

    def singlet(self, i, j):
        """
        operator definition is taken from here
        http://dx.doi.org/10.1016/j.jmr.2015.08.021
        !! Currently only applicable for a manifold of 1/2 spins !!
        
        Parameters
        ----------
        i : int
            Idx of the first spin
        j : int
            Idx of the second spins

        Returns
        -------
        numpy.ndarray
            The operator we asked for
        """       
        return self.scalar(i, j, secular=False) * 2**(2 - self.get_n_spins())
    
    def pol_level(self, i, label="z"):
        """
        operator definition is taken from here
        http://dx.doi.org/10.1016/j.jmr.2015.08.021
        !! Currently only applicable for a manifold of 1/2 spins !!
        
        Parameters
        ----------
        i : int
            Idx of the first spin
        j : int
            Idx of the second spins

        Returns
        -------
        numpy.ndarray
            The operator we asked for
        """       
        return self.op(i, label) * 2 ** (1 - self.get_n_spins())
    
    def set_lamour_freq(self, field):
        # get the frequencies
        self.lamour_freq = {i: field * sd.gamma(i) for i in set(self.nuclei_list)}
        self.lamour_freq_hz = {
            i: freq / (np.pi * 2) for i, freq in self.lamour_freq.items()
        }

    def get_freq_shift(self, atom_id, ppm, absolute=False, freq="hz"):
        atom_type = self.nuclei_list[atom_id]
        lamour_val = (
            self.lamour_freq_hz[atom_type]
            if freq == "hz"
            else self.lamour_freq[atom_type]
        )
        shift = lamour_val * ppm
        return shift if not absolute else lamour_val + shift

    def get_shift_hz(self, atom_id, ppm, absolute=False):
        """A wrapper for get_freq_shift that always gives the frequency in hertz

        Parameters
        ----------

        atom_id : int
            The atom ID is an int that goes from 0 to len(System.nuclei_list)
        ppm : float
            chemical shift
        absolute : bool, optional
            If True, it returns the absolute frequency. If False, it returns the frequency relative to the carrier

        Returns
        -------
        float
            The frequency in hertz
        """
        return self.get_freq_shift(atom_id, ppm, absolute=absolute, freq="hz")

    def get_shift_omega(self, atom_id, ppm, absolute=False):
        """A wrapper for get_freq_shift that always gives the frequency in hertz

        Parameters
        ----------

        atom_id : int
            The atom ID is an int that goes from 0-len(System.nuclei_list)
        ppm : float
            chemical shift
        absolute : bool, optional
            If True it returns the absolute frequency. If False it returns the frequency relative to the carrier

        Returns
        -------
        float
            The frequency in angular frequency
        """
        return self.get_freq_shift(atom_id, ppm, absolute=absolute, freq="omega")


        
    def print_rho(
        self,
    ):
        """Summary"""
        print(np.array_str(self.rho, precision=2, suppress_small=True))

    def calc_single_spin_rotation(self, phase, angle):
        c, s = np.cos(angle / 2), np.sin(angle / 2)
        if phase == "x":
            return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.cdouble)
        elif phase == "y":
            return np.array([[c, -s], [s, c]], dtype=np.cdouble)
        elif phase == "z":
            return np.array([[c - 1j * s, 0], [0, c + 1j * s]], dtype=np.cdouble)
        else:
            print("Phase for rotation is not recognised")

    def calc_rotation_mat(self, phase, angle, active="all"):
        # this is the rotation matrix
        rotation = self.calc_single_spin_rotation(phase, angle)

        if active == "all":
            res = 1
            for _ in range(self.n_spins):
                res = self.kron(res, rotation)

        # apply the rotation to onle the active spins
        else:
            res = 1
            for i in range(self.n_spins):
                if i in active:
                    res = self.kron(res, rotation)
                else:
                    res = self.kron(res, self.small_identity)
                
        return res

    def apply_rotation(self, phase, angle, active="all"):
        """Applies the rotation to the density matrix System.rho

        Parameters
        ----------
        phase : str
            Phase of the rotation. It can be 'x', 'y', 'z'
        angle : float
            The angle by which we want to to rotate the active spins
        active : str, list
            If 'all' the rotation matrix is applied to all atoms. If a list is given then
            the rotation is only applied to the atom_ids present in the list

        """
        rotation = self.calc_rotation_mat(phase, angle, active=active)
        rotation_inv = rotation.conj().T
        self.rho = functools.reduce(
            self.matmul,
            [rotation, self.rho, rotation_inv]
        )

    def apply_hamiltonian(self, hamiltonian, time, rho):
        pre_operator = self.expm(-1j * hamiltonian * time)
        post_operator = pre_operator.conj().T
        return functools.reduce(
            self.matmul, 
            [pre_operator, rho, post_operator]
        )

    def apply_hamiltonian_to_self(self, hamiltonian, time):
        """Takes the Hamiltonian and applies it to the density matrix System.rho

        Parameters
        ----------
        hamiltonian : numpy.ndarray
            The Hamiltonian you want to evolve
        time : float
            time in seconds
        """
        self.rho = self.apply_hamiltonian(hamiltonian, time, self.rho)

    def get_trace(
        self,
    ):
        """Returns the trace of System.rho"""
        return np.trace(self.rho)

    def scalar_coupling_hamil(self, spin1, spin2, coupling):
        """determines the scalar coupling Hamiltonian for two spins

        Parameters
        ----------
        spin1 : int
            atom_id for first spin
        spin2 : int
            atom_id for second spin
        coupling : float
            scalar coupling in hz

        Returns
        -------
        numpy.ndarray
            the scalar coupling Hamiltonian for the two spins
        """
        # we don't need to consider strong coupling if we have different nuclei types
        # If you really wanted you could also add a check for the shift difference vs coupling too
        if self.nuclei_list[spin1] == self.nuclei_list[spin2]:
            axis = ["x", "y", "z"]
        else:
            axis = ["z"]

        i_operators = [self.op_full(f"{spin1}{i}") for i in axis]
        s_operators = [self.op_full(f"{spin2}{i}") for i in axis]

        operator_products = 0
        for i, j in zip(i_operators, s_operators):
            operator_products += self.matmul(i, j)
        
        total = coupling * operator_products * 2 * np.pi
        
        return total

    def calc_fid(self, hamiltonian, time_array, detection_operators="all"):
        """calculates the free induction decay with a given Hamiltonian. Currently
        in this implementation it is assumed that the Hamiltonian is time independent.

        Parameters
        ----------
        hamiltonian : numpy.ndarray
            An array that has the Hamiltonian with which you want to evolve the system
        time_array : numpy.ndarray
            an array with the time points you want to calculate the free induction decay over
        detection_operators : str, list
            If all the detection operator I+ is used for all the spins in the system. Otherwise
            give a list of the individual operators you would like to sum together to detect with.

        Returns
        -------
        time_array : numpy.ndarray
            the time array used for the evolution
        fid : numpy.ndarray
            the free induction decay
        """
        # do some setup
        fid = []
        if detection_operators == "all":
            ops = 0
            for i in range(self.n_spins):
                ops += self.op_full(f"{i}p")

        else:
            ops = 0
            for i in detection_operators:
                ops += self.op_full(i)
            

        # the domain we will calculate
        dtime = time_array[1] - time_array[0]
        points = len(time_array)

        # set up the sparse matricies, could remove this part if we
        # move to only using sparse matricies, this would tidy things up generally
        sparse_rho = sps.csc_matrix(self.rho)
        sparse_hamil = sps.csc_matrix(hamiltonian)
        sparse_detection_operator = sps.csc_matrix(ops)

        # calculate the propergator
        prop_part = sps.linalg.expm(-1j * sparse_hamil * dtime)
        prop_part_inv = sps.linalg.inv(prop_part)

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

def sparse_dot(A, B):
    """This is a helper function to provide two arguments multiplication function for sparse matrices

    Parameters
    ----------
    A : scipy sparse matrix
    B : scipy sparse matrix

    Returns
    -------
    scipy sparse matrix
    """
    return A.dot(B)