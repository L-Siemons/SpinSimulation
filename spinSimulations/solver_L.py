### HERE SOLVERS FOR LIOUVILLIAN FORMALISM WILL BE
import numpy as np
import scipy as sp
import scipy.sparse as sps
from spinSimulations.solver_H import _set_ops

# ALL VECTORIZATION ARE PERFORMED ROWWISE
def mat2vec(rho):
    if sps.isspmatrix(rho):
        return _mat2vec_sparse(rho)
    else:
        return _mat2vec_dense(rho)

def _mat2vec_sparse(rho):
    if not sps.isspmatrix_csr(rho):
        rho = rho.tocsr()
    return rho.reshape((1, np.prod(rho.shape)), order='C')
    
def _mat2vec_dense(rho):
    return rho.reshape(np.prod(rho.shape), order='C')

def _set_ops(op, system=None):
    # matmul, elem_mul, kron, expm, eye
    if system:
        return system.matmul, system.multiply, system.kron, system.expm, system.eye
    
    if sps.issparse(op):
        return sparse_dot, sparse_multiply, sps.kron, sps.linalg.expm, sps.eye
    else:
        return np.matmul, np.multiply, np.kron, sp.linalg.expm, np.eye
    
def prop_uni_SO(op, prop):
    matmul, _, _, _, _ = _set_ops(op)
    return matmul(mat2vec(op), prop)

def vec2mat(rho):
    if sps.isspmatrix(rho):
        return _vec2mat_sparse(rho)
    else:
        return _vec2mat_dense(rho)

def _vec2mat_sparse(rho):
    if not sps.isspmatrix_csr(rho):
        rho = rho.tocsr()
    N=int(rho.shape[1]**0.5)
    return rho.reshape(N, N)
    
def _vec2mat_dense(rho):
    N=int(rho.shape[0]**0.5)
    return rho.reshape(N, N)

def amplitude_v(A, B):
    return np.real(np.dot(A,B)/np.dot(A,A))

def calc_uni_SO(op, duration=None, spin_system=None):
    _, _, _, expm, _ = _set_ops(op, spin_system)
    if duration is None:
        return expm(-1j * comm_SO(op))
    return expm(-1j * comm_SO(op) * duration)

def projector_SO(A):
    return np.outer(mat2vec(A), mat2vec(A).conj()) / np.dot(mat2vec(A), mat2vec(A).conj())

def leftm_SO(A, spin_system=None):
    _, _, kron, _, eye = _set_ops(A, spin_system)
    return kron(A, eye(A.shape[0]))

def rightm_SO(A, spin_system=None):
    _, _, kron, _, eye = _set_ops(A, spin_system)
    return kron(eye(A.shape[0]),A.T)

def comm_SO(A, spin_system=None):
    return leftm_SO(A,spin_system) - rightm_SO(A,spin_system)

# TODO: it is a replica from a spin file, do something with it
def sparse_dot(a, b):
    """This is a helper function to provide two arguments multiplication function for sparse matrices

    Parameters
    ----------
    a : scipy sparse matrix
    b : scipy sparse matrix

    Returns
    -------
    scipy sparse matrix
    """
    return a.dot(b)

# TODO: the same as sparse dot
def sparse_multiply(a, b):
    return a.multiply(b)

###SABRE superoperators
def kron_SO(ss_A, ss_AB, rho):
    """
        Direct product between the states of A and B subsystems
        Corresponds to the substrate association with the complex
        Reaction: (A) + B -> AB
        where B is a pH2 molecule
    """
    dim_B = 4
    I_A,I_B=np.eye(ss_A.get_spin_dim()),np.eye(dim_B)
    return np.einsum("ix,jk,lX,mn,xX->ijlmkn",I_B,I_A,I_B,I_A,rho).reshape((ss_AB.get_spin_dim())**2,(ss_A.get_spin_dim())**2)

def part_trace_SO(ss_A, ss_AB):
    """
        Partial trace over the states of the subsystem B
        Corresponds to the substrate dissociation from the complex
        Reaction: AB -> (A) + B
        where B is a pH2 molecule
    """
    dim_B = 4
    I_A,I_B=np.eye(ss_A.get_spin_dim()),np.eye(dim_B)
    return np.einsum("xi,jk,xl,mn->jmikln",I_B,I_A,I_B,I_A).reshape((ss_A.get_spin_dim())**2,(ss_AB.get_spin_dim())**2)

def relax_phenom_SO(ss, T1, T2 = None):
    """
    T1 and T2 relaxation uitilizing the projectors on the corresponding spin operators
    """
    if T2 == None:
        T2 = T1
    R = 0
    for idx in range(ss.n_spins):
        R += -1/T1[idx] * projector_SO(ss.op(idx, 'z'))
        R += -1/T2[idx] * projector_SO(ss.op(idx, 'p'))
        R += -1/T2[idx] * projector_SO(ss.op(idx, 'm')) 
    return R