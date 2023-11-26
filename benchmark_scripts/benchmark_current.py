import spinSimulations.spin as s
import spinSimulations.utils_spectra as spec_utils
import importlib
import matplotlib.pyplot as plt
import numpy as np
import random as r
from timeit import default_timer as timer

file_name = "sparse_matricies.dat"

lines = []
for i in range(2, 10):
    start_time = timer()
    nspin = i
    atom_list = ["1H"] * nspin

    system = s.System(nspin, atom_list)
    start = np.sum([system.operator(f"{i}z") for i in range(nspin)], axis=0)
    system.rho = start
    system.load_gammas()
    system.set_lamour_freq(14)

    number_of_points = 1000
    time_array = np.linspace(0, 1, number_of_points)
    dtime = time_array[1] - time_array[0]
    detection_operator = [f"{i}p" for i in range(nspin)]

    chemical_freqs = []
    chemical_freqs_hamils = []
    couplings = []
    coupling_hamils = []

    for i in range(nspin):
        curr_shift_ppm = r.uniform(-0.1, 1) * 1e-6
        curr_shift_hz = system.get_freq_shift(i, curr_shift_ppm)
        chemical_freqs.append(curr_shift_hz)
        chemical_freqs_hamils.append(
            np.pi * 2 * curr_shift_hz * system.operator(f"{i}z")
        )

    zero_threshold = 0.7
    for i in range(nspin):
        for j in range(i + 1, nspin):
            if r.uniform(0, 1) > zero_threshold:
                current_coupling = r.uniform(0, 30)
                couplings.append([i, j, current_coupling])
                coupling_hamils.append(
                    system.scalar_coupling_hamil(i, j, current_coupling)
                )

    j_hamil = np.sum(coupling_hamils, axis=0)
    cs_hamil = np.sum(chemical_freqs_hamils, axis=0)
    total_hamil = j_hamil + cs_hamil

    system.apply_rotation("y", np.pi / 2)
    time_axis, fid = system.calc_fid(
        total_hamil, time_array, detection_operators=detection_operator
    )

    fid = fid * np.e ** (-15 * time_array)
    axis, ft = spec_utils.organise_1d(fid, number_of_points, dtime)
    end_time = timer()

    line = f"spins {nspin} time: {end_time - start_time:0.05} s"
    print(line)  # Time in seconds, e.g. 5.38091952400282
    lines.append(line)

f = open(file_name, "w")
f.write("\n".join(lines))
f.close()
