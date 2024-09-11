### HERE SOLVERS FOR LIOUVILLIAN FORMALISM WILL BE
import numpy as np
import scipy.sparse as sps

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

def vec2mat(rho):
    N=int(np.shape(rho)[0]**0.5)
    return rho.reshape(N,N)

def projector_SO(A):
    return np.outer(mat2vec(A), mat2vec(A).conj()) / np.dot(mat2vec(A), mat2vec(A).conj())

def leftm_SO(A):
    return np.kron(A,np.eye(A.shape[0]))

def rightm_SO(A):
    return np.kron(np.eye(A.shape[0]),A.T)

def comm_SO(A):
    return leftm_SO(A) - rightm_SO(A)

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