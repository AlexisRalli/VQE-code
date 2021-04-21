from scipy.sparse.linalg import expm
from openfermion import qubit_operator_sparse
from scipy.sparse import find
from scipy.sparse import csr_matrix
import numpy as np
from copy import deepcopy
from functools import reduce

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




from difflib import SequenceMatcher
def string_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

from openfermion import QubitOperator
from openfermion.utils import count_qubits
from functools import reduce


def lexicographical_sort(P_words):
    fullOp = reduce(lambda Op1, Op2: Op1+Op2, P_words)
    max_qubits = count_qubits(fullOp)

    P_Words = []
    for op in P_words:
        Q_Nos, P_strings = zip(*list(*op.terms.keys()))
        
        arr = [f'{P_strings[Q_Nos.index(qNo)]}{qNo}' if qNo in Q_Nos else f'I{qNo}' for qNo in range(max_qubits)]
          
        P_Words.append(' '.join(arr))
    
    P_Words_copy = deepcopy(P_Words)
    re_ordered_ind =[]
    sorted_list = []
    while P_Words!=[]:
        if sorted_list==[]:
            ind_match=0
        else:
            op_prev = sorted_list[-1] # take last sorted term
            
            similarity_list = [(op_j, string_similarity(op_prev, op_j)) for op_j in P_Words if op_j != op_i]
            largest_match = max(similarity_list, key=lambda x:x[1])
            ind_similarity_list = similarity_list.index(largest_match)

            op_j = similarity_list[ind_similarity_list][0]
            ind_match = P_Words.index(op_j)
            
        op_i = P_Words.pop(ind_match)
        sorted_list.append(op_i)
        re_ordered_ind.append(P_Words_copy.index(op_i))


    lex_sorted = (np.array(P_words)[re_ordered_ind]).tolist()
    return lex_sorted


def choose_Pn_index(AC_set_list):
    """
    given a list of anti-commuting operators

    Return index of term with fewest number of change of basis required and fewest Z count

    """
    
    # # size sorting doesn't matter here, as not ansatz or Hamiltonian!
    # sorted_by_size = sorted(AC_set_list, key = lambda x: len(list(zip(*list(*x.terms.keys())))[1]))

   
    ## minimize change of basis
    for ind, op in enumerate(AC_set_list):
        if ind ==0:
            best_ind = 0
            best_Qno, best_Pstrings = zip(*list(*op.terms.keys()))
            best_N_change_basis = sum([1 for sig in best_Pstrings if sig != 'Z'])
            best_Z_count = sum([1 for sig in best_Pstrings if sig == 'Z'])
        else:
            
            Q_Nos, P_strings = zip(*list(*op.terms.keys()))
            N_change_basis = sum([1 for sig in P_strings if sig != 'Z'])

            if (N_change_basis<=best_N_change_basis):
                best_ind = deepcopy(ind)
                best_N_change_basis = deepcopy(N_change_basis)
                best_Z_count = sum([1 for sig in P_strings if sig == 'Z'])
    
     
    ## minimize number of Z terms
    for ind, op in enumerate(AC_set_list):   
        Q_Nos, P_strings = zip(*list(*op.terms.keys()))
        Z_count = sum([1 for sig in P_strings if sig == 'Z'])
        N_change_basis = sum([1 for sig in P_strings if sig != 'Z'])
        
        if (N_change_basis==best_N_change_basis) and (Z_count<best_Z_count): # note keeps best change of basis!
            best_ind = deepcopy(ind)
            best_N_change_basis = deepcopy(N_change_basis)
            best_Z_count= deepcopy(Z_count)
    return  best_ind