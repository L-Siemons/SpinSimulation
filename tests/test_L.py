from random import shuffle
import numpy as np
import scipy.sparse as sps
import spinSimulations.spin as s
from spinSimulations.solver_L import mat2vec

# TODO: these are simple tests
# General test case should be written
class TestVectorization:
    
    def test_mat2vec_dense(self):
        for n in range(1, 10):
            my_list = list(range(n**2))
            shuffle(my_list)
            my_list_np = np.array(my_list)
            my_array_np = my_list_np.copy().reshape((n, n))
            assert np.array_equal(
                my_list_np,
                mat2vec(my_array_np)
            )
            
    def test_mat2vec_sparse(self):
        for n in range(2, 10):
            my_list = list(range(n**2))
            shuffle(my_list)
            my_list_np = np.array(my_list)
            my_array_np = sps.csr_matrix(my_list_np.copy().reshape((n, n)))
            assert np.array_equal(
                my_list_np,
                mat2vec(my_array_np).toarray()[0]
            )


if __name__ == "__main__":
    for n in range(2, 4):
        my_list = list(range(n**2))
        shuffle(my_list)
        my_list_np = np.array(my_list)
        my_array_np = sps.csr_matrix(my_list_np.copy().reshape((n, n)))
        print(sps.csr_matrix(my_list_np).toarray())
        print(mat2vec(my_array_np).toarray().astype(my_list_np.dtype))
        print('=====')