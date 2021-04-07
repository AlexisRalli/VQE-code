from scipy.sparse.linalg import expm
from openfermion import qubit_operator_sparse
from scipy.sparse import find
from scipy.sparse import csr_matrix
import numpy as np

def sparse_allclose(A, B, atol=1e-8, rtol=1e-05):
    # https://stackoverflow.com/questions/47770906/how-to-test-if-two-sparse-arrays-are-almost-equal

    if np.array_equal(A.shape, B.shape) == 0:
        raise ValueError('Matrices different shapes!')

    r1, c1, v1 = find(A)  # row indices, column indices, and values of the nonzero matrix entries
    r2, c2, v2 = find(B)

    # # take all the important rows and columns
    # rows = np.union1d(r1, r2)
    # columns = np.union1d(c1, c2)

    # A_check = A[rows, columns]
    # B_check = B[rows, columns]
    # return np.allclose(A_check, B_check, atol=atol, rtol=rtol)

    compare_A_indces = np.allclose(A[r1,c1], B[r1,c1], atol=atol, rtol=rtol)
    if compare_A_indces is False:
        return False

    compare_B_indces = np.allclose(A[r2,c2], B[r2,c2], atol=atol, rtol=rtol)

    if (compare_A_indces is True) and (compare_B_indces is True):
        return True
    else:
        return False