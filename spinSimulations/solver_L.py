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