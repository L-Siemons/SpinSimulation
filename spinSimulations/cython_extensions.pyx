
import numpy as np 
import math
import cython
cimport numpy as np
import scipy

def kron_all_v2(list array_list):

    cdef int num_arrays = len(array_list)
    cdef np.ndarray[np.complex128_t, ndim=2] result = array_list[0]

    for i in range(1, num_arrays):
        result = np.kron(result, array_list[i])

    return result

# def matrix_exp(A):
#     '''
#     This function calculates a matrix exponencial. In the simulation of the
#     relaxometry intensities this is the slowest step and should be optimised the 
#     most. The implimentation here should give the same results as scipy.lingalg.expm.

#     Currently this is done with numpy and Cython however I am open to other
#     alternatives.

#     Parameters
#     ----------
#         A :  np.ndarray[np.float64_t, ndim=2]

#     Returns
#     -------
#         matrix_exp : np.ndarray[np.float64_t, ndim=2]
#     '''

#     return  scipy.linalg.expm(A)

def matrix_exp( np.ndarray[np.complex128_t, ndim=2] A):
    '''
    This function calculates a matrix exponencial. In the simulation of the
    relaxometry intensities this is the slowest step and should be optimised the 
    most. The implimentation here should give the same results as scipy.lingalg.expm.

    Currently this is done with numpy and Cython however I am open to other
    alternatives.

    Parameters
    ----------
        A :  np.ndarray[np.float64_t, ndim=2]

    Returns
    -------
        matrix_exp : np.ndarray[np.float64_t, ndim=2]
    '''

    cdef np.ndarray[np.complex128_t, ndim=1] eigenvalues
    cdef np.ndarray[np.complex128_t, ndim=2] eigenvectors
    cdef np.ndarray[np.complex128_t, ndim=2] diagonal_matrix
    cdef np.ndarray[np.complex128_t, ndim=2] matrix_exp
    cdef np.ndarray[np.complex128_t, ndim=2] intermediate

    eigenvalues, eigenvectors = np.linalg.eig(A)
    diagonal_matrix = np.diag(np.exp(eigenvalues))
    intermediate = np.matmul(eigenvectors, diagonal_matrix)
    matrix_exp = np.matmul(intermediate, np.linalg.inv(eigenvectors))
    return matrix_exp #scilinalg.expm(A) #