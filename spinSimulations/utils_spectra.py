"""somewhere to keep utility functions
"""
import numpy as np 

def get_axis(number_of_points, dtime):
	"""determines the frequency axis of a spectrum 
	
	Parameters
	----------
	number_of_points : int
	    number of points in the spectrum
	dtime : float
	    time incriment between points
	
	Returns
	-------
	numpy.ndarray
	    the frequency axis
	"""
	return np.fft.fftfreq(number_of_points, d=dtime)

def organise_1d(fid, number_of_points, dtime):
	"""Gets the frequency axis for a spectrum, does the fourier transform in 1D and also 
	applies the fft shift. 
	
	Parameters
	----------
	fid : numpy.ndarray
	    The free induction decay
	number_of_points : int
	    number of points in the spectrum
	dtime : float
	    time incriment between points
	
	Returns
	-------
	axis : numpy.ndarray
	    The frequency axis
	ft : numpy.ndarray
	    The transformed spectrum
	"""
	ft = np.fft.fft(fid)
	ft = np.fft.fftshift(ft)
	axis = get_axis(number_of_points, dtime)
	axis = np.fft.fftshift(axis)
	return axis, ft