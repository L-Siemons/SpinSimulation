"""Contains the main class System which hold information and 
opperations about the spin system
"""
# this module 
import spinSimulations.cython_extensions as cext

#numerical modules
import scipy 
import scipy.sparse as sparse 
import numpy as np 

# additional functionality
import re 
import functools
import copy
import pkg_resources



class System():

    """Summary
    
    Attributes
    ----------
    atom_types : list
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
        Lowering operator in the raising/lowering basis for a sigle spin
    ip0 : numpy.ndarray
        Raising operator in the raising/lowering basis for a sigle spin
    ix0 : numpy.ndarray
        Ix operator in the cartesian basis for a sigle spin
    iy0 : numpy.ndarray
        Iy operator in the cartesian basis for a sigle spin
    iz0 : numpy.ndarray
        Iz operator in the cartesian basis for a sigle spin
    lamour_freq : dict
        This contains all the lamour frequencies at a given magnetic field (T)
    lamour_freq_hz : dict
        This contains all the lamour frequencies at a given magnetic field (T) in hz
    nspins : int
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
    
    def __init__(self, nspins, atom_types):
        '''
        Initial set up 
        
        Parameters
        ----------
        nspins : int
            number of spins in the system
        atom_types : list
            A list of atom types in the spin system. Each position is associated with an atom ID, 
            ie the position in this list, which is used to identify each atom in the system. This list 
            is also used to determine gyromagnetic ratio. All the available atom types are listed in
            spinSimulations/dat/gammas.dat
        '''
        self.ix0 = np.array([[0,1],[1,0]], dtype=np.cdouble) * 0.5 
        self.iy0 = np.array([[0,-1j],[1j,0]], dtype=np.cdouble) * 0.5
        self.iz0 = np.array([[1,0],[0,-1]], dtype=np.cdouble) * 0.5 
        
        # raising and lowering opperators
        self.ip0 = self.ix0 + 1j * self.iy0
        self.im0 = self.ix0 - 1j * self.iy0
        self.i00 = self.iz0 * np.sqrt(2)
        
        self.single_spin_dictionary = {}
        self.single_spin_dictionary['x'] = self.ix0
        self.single_spin_dictionary['y'] = self.iy0
        self.single_spin_dictionary['z'] = self.iz0
        
        self.single_spin_dictionary['p'] = self.ip0
        self.single_spin_dictionary['m'] = self.im0
        self.single_spin_dictionary['o'] = self.i00
        
        self.small_identity = np.identity(2).astype(np.cdouble)
        self.operators = {}
        self.nspins = nspins
        self.rho = None

        self.gammas = None
        self.carriers_hz = None
        self.carriers_omega = None

        assertion_message = 'There are different numbers of entries in nspins and atom_types'
        assert nspins == len(atom_types), assertion_message
        self.atom_types = atom_types

    def load_gammas(self):
        """Loads the gyromagnetic ratios
        """
        self.gammas_file = pkg_resources.resource_filename('spinSimulations', 'dat/gammas.dat')

        self.gammas = {}
        f = open(self.gammas_file)
        for line in f.readlines():
            split = line.split()
            check = True
            if len(split) !=2:
                check = False
            if line[0] == '#':
                check = False

            if check == True:
                self.gammas[split[0]] = float(split[1])

        f.close()
        #print(self.gammas)

    def set_lamour_freq(self, field):
        """Sets the lamour frequencies
        
        Parameters
        ----------
        field : float 
            The field in Tesla
        """
        # might need to load the gammas :) 
        if self.gammas == None:
            self.load_gammas()
        
        # now we calculate all the carriers
        self.lamour_freq = {}
        self.lamour_freq_hz = {}
        
        for i in set(self.atom_types):

            # check the atoms names exist
            assertion_message = 'please ensure that the atom types are as in:\n {self.gammas_file}'
            assert i in self.gammas, assertion_message

            # get the frequencies
            self.lamour_freq[i] = field * self.gammas[i]
            self.lamour_freq_hz[i] =  self.lamour_freq[i]/(np.pi*2)

    # the naming here is missleading. 
    def get_freq_shift(self, atom_id, ppm, absolute=False, freq='hz'):
        """convers the chemical shift of an atom to the frequency in hz
        
        Parameters
        ----------
        atom_id : int
            The atom ID is an int that goes from 0-len(System.atom_types)
        ppm : float
            chemical shift
        absolute : bool, optional
            If True it returns the abosolute frequency. If False it returns the frequency relative to the carrier
        freq : str, optional
            Takes the options 'hz' or  'omega' depending on which units are needed.
        
        Returns
        -------
        float
            The frequency in hertz or angular frequencies depending on the value of freq
        """
        atom_type = self.atom_types[atom_id]

        if freq == 'hz':
            lamour_val = self.lamour_freq_hz[atom_type]
        
        elif freq == 'omega':
            lamour_val = self.lamour_freq[atom_type]
        
        freq = lamour_val * ppm

        # distance from the lamour frequency
        if absolute == False:
            return freq

        #absolute frequency
        if absolute == True:
            return lamour_val + freq

    def get_shift_hz(self, atom_id, ppm, absolute=False):
        """A wrapper for get_freq_shift that always gives the frequency in hertz
        
        Parameters
        ----------

        atom_id : int
            The atom ID is an int that goes from 0-len(System.atom_types)
        ppm : float
            chemical shift
        absolute : bool, optional
            If True it returns the abosolute frequency. If False it returns the frequency relative to the carrier
        
        Returns
        -------
        float
            The frequency in hertz
        """
        return self.get_freq_shift(atom_id, ppm, absolute=absolute, freq='hz')

    def get_shift_omega(self, atom_id, ppm, absolute=False):
        """A wrapper for get_freq_shift that always gives the frequency in hertz
        
        Parameters
        ----------

        atom_id : int
            The atom ID is an int that goes from 0-len(System.atom_types)
        ppm : float
            chemical shift
        absolute : bool, optional
            If True it returns the abosolute frequency. If False it returns the frequency relative to the carrier
        
        Returns
        -------
        float
            The frequency in angular frequency
        """
        return self.get_freq_shift(atom_id, ppm, absolute=absolute, freq='omega')

    def kron_all(self, array_list):
        """Applied the kroneker product to all arrays in array_list. This is used to make the 
        operators in a multi spin system
        
        Parameters
        ----------
        array_list : list
            list of 2d numpy.ndarray
        
        Returns
        -------
        numpy.ndarray
            the resulting matrix of all the kroneker products
        """
        return cext.kron_all_v2(array_list)

    def operator(self,name):
        '''
        Get the operator if is pressent in System.operators. Otherwise it is generated and added to the
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
        '''

        if name in self.operators:
            return self.operators[name]

        else:
            operator_str = name.split('.')
            operator_str = [re.findall(r'\D+|\d+', a) for a in operator_str]
            operator_str = np.array(operator_str, dtype=str).flatten()
            
            new_shape = (len(operator_str)//2,2)
            if new_shape[0] == 0:
                new_shape[0] = 1
            
            operator_str = np.reshape(operator_str, new_shape)
            operator_list = [self.small_identity for i in range(self.nspins)]

            for i in operator_str:
                counter = int(i[0])
                operator_list[counter] = self.single_spin_dictionary[i[1]]
                
            self.operators[name] = self.kron_all(operator_list)
            return self.operators[name]
        
    def print_rho(self,):
         """Summary
         """
         print(np.array_str(self.rho, precision=2, suppress_small=True))
            
    def calc_single_spin_rotation(self, phase, angle):
        """calculates the rotation matrix for a single spin
        
        Parameters
        ----------
        phase : str
            Phase of the rotation. It can be 'x', 'y', 'z'
        angle : float
            The angle by which we want to to rotate the system
        
        Returns
        -------
        numpy.ndarray
            the rotation matrix
        """
        c = np.cos(angle/2)
        s = np.sin(angle/2)
        
        if phase == 'x':
            return np.array([[c, -1j*s],[-1j*s, c]], dtype=np.cdouble)
        elif phase == 'y':
            return np.array([[c, -s],[s, c]], dtype=np.cdouble)
        elif phase == 'z':
            return np.array([[c-1j*s, 0],[0, c+1j*s]], dtype=np.cdouble)
        else:
            print('Phase for rotation is not recognised')
            
    def calc_rotation_mat(self, phase, angle, active='all'):
        """Calculates the rotation matrix for the total system
        
        Parameters
        ----------
        phase : str
            Phase of the rotation. It can be 'x', 'y', 'z'
        angle : float
            The angle by which we want to to rotate the active spins
        active : str, list
            If 'all' the rotation matrix is applied to all atoms. If a list is given then 
            the rotation is only applied to the atom_ids present in the list           
        
        Returns
        -------
        numpy.ndarray
            The final roation matrix
        """
        #this is the rotation matrix
        rotation = self.calc_single_spin_rotation(phase, angle)
        
        #apply the rotation to all matricies
        if active == 'all':
            rotation_list = [rotation for _ in range(self.nspins)]
            return self.kron_all(rotation_list)
        
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
        rotation_inv = np.linalg.inv(rotation)
        #print([rotation, self.rho, rotation_inv])
        self.rho = functools.reduce(np.matmul, [rotation, self.rho, rotation_inv])
    
    def apply_hamiltonian(self, hamiltonian, time, rho):
        """Takes the hamiltonian and applies it to the density matrix rho
        
        Parameters
        ----------
        hamiltonian : numpy.ndarray
            The hamiltonian you want to evolve
        time : float
            time in seconds
        rho : numpy.ndarray
            The density matrix
        
        Returns
        -------
        numpy.ndarray
            The density matrix after evolving the system
        """
        pre_operator = cext.matrix_exp(-1j*hamiltonian*time)
        post_operator = np.linalg.inv(pre_operator)
        #post_operator = scipy.linalg.expm(1j*hamiltonian*time)
        return functools.reduce(np.matmul, [pre_operator, rho, post_operator])
    
    def apply_hamiltonian_to_self(self, hamiltonian, time):
        """Takes the hamiltonian and applies it to the density matrix System.rho
        
        Parameters
        ----------
        hamiltonian : numpy.ndarray
            The hamiltonian you want to evolve
        time : float
            time in seconds
        """
        self.rho = self.apply_hamiltonian(hamiltonian, time, self.rho)

    def get_trace(self,):
        """Returns the trace of System.rho"""
        return np.trace(self.rho)
    
    def scalar_coupling_hamil(self, spin1, spin2, coupling):
        """determines the scalar coupling hamiltonian for two spins 
        
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
            the scalar coupling hamiltonian for the two spins
        """
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
        """calculates the free induction decay with a given hamiltonian. Currently 
        in this implimentation it is assumed that the hamiltonian is time independant. 
        
        Parameters
        ----------
        hamiltonian : numpy.ndarray
            An array that has the hamiltonian with which you want to evolve the system
        time_array : numpy.ndarray
            an array with the time points you want to calculate the free induction decay over
        detection_operators : str, list
            If all the detection operator I+ is used for all the spins in the system. Otherwise
            give a list of the individual operators you would like to sum togeter to detect with. 
        
        Returns
        -------
        time_array : numpy.ndarray
            the time array used for the evolution
        fid : numpy.ndarray
            the free induction decay
        """
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