import spinSimulations.spin as s
import spinSimulations.utils_spectra as spec_utils

import matplotlib.pyplot as plt
import numpy as np 
import random as r
from timeit import default_timer as timer


start_time = timer()

# set the number of spins, for nspin>6 it will take time ...
nspin = 5
# set the spin types, here all proton
atom_list = ['1H']*nspin

#number of points in the spectrum
number_of_points = 1000

# time axis for the FID
time_array = np.linspace(0, 1, number_of_points)
dtime = time_array[1]-time_array[0]

# set up the system
system = s.System(nspin,  atom_list)

# define the initial operators, here all spins are on Z
start = np.sum([system.op_full(f'{i}z') for i in range(nspin)], axis=0)

#set rho
system.rho = start 
system.load_gammas()
system.set_lamour_freq(14)

# the operator we use for detection
detection_operator = [f'{i}p' for i in range(nspin)]

# here I determine the chemical shift and scalar coupling 
# hamiltonians 
chemical_freqs = []
chemical_freqs_hamils = []
couplings = []
coupling_hamils = []

# shifts
for i in range(nspin):
    curr_shift_ppm = r.uniform(-0.1, 1)*1e-6
    curr_shift_hz = system.get_freq_shift(i, curr_shift_ppm)
    chemical_freqs.append(curr_shift_hz)
    chemical_freqs_hamils.append(np.pi*2*curr_shift_hz*system.op_full(f'{i}z'))


# couplings
zero_threshold = 0.6
for i in range(nspin):
    for j in range(i+1,nspin):
        
        if r.uniform(0,1) > zero_threshold:
            current_coupling = r.uniform(0,30)
            couplings.append([i,j, current_coupling])
            coupling_hamils.append(system.scalar_coupling_hamil(i,j,current_coupling))

#add them together
j_hamil = np.sum(coupling_hamils, axis=0)
cs_hamil = np.sum(chemical_freqs_hamils, axis=0)
total_hamil = j_hamil + cs_hamil

# apply a 90 degree pulse
system.apply_rotation('y', np.pi/2 )

#aquire
time_axis, fid = system.calc_fid(total_hamil, time_array, detection_operators=detection_operator)

# add some rough relaxation
fid =  fid*np.e**(-5*time_array)

# re-organise the spectrum and do FT
axis, ft = spec_utils.organise_1d(fid, number_of_points, dtime)

#timer 
end_time = timer()
print('Total time taken', end_time - start_time) # Time in seconds, e.g. 5.38091952400282


#plot
plt.plot(time_axis, np.real(fid))
plt.xlabel('time')
plt.show()

plt.plot(axis, np.real(ft))
plt.xlabel('frequency')
plt.show()
