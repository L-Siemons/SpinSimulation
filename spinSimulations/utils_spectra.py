
import numpy as np 

def get_axis(number_of_points, dtime):
	return np.fft.fftfreq(number_of_points, d=dtime)

def organise_1d(fid, number_of_points, dtime):

	ft = np.fft.fft(fid)
	ft = np.fft.fftshift(ft)
	axis = get_axis(number_of_points, dtime)
	axis = np.fft.fftshift(axis)
	return axis, ft