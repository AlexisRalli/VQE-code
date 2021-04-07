import numpy as np
import cs_vqe as c

import ast
import matplotlib.pyplot as plt
import pickle

from quchem.Unitary_partitioning_Seq_Rot import *
from quchem.Unitary_partitioning_LCU_method import * 
from scipy.sparse.linalg import expm
from openfermion import qubit_operator_sparse
from copy import deepcopy as copy
import pprint

with open("hamiltonians.txt", 'r') as input_file:
    hamiltonians = ast.literal_eval(input_file.read())

## LOAD data
file_name = 'anticommuting_vs_standard_conH.pickle'
with open(file_name, 'rb') as infile:
    loaded_dict = pickle.load(infile)
    
# E_LCU_dict ={}

    
# N_Qubits= hamiltonians['H1-Li1_STO-3G_singlet'][1]

# qubit_ordering=[i for i in range(N_Qubits)]
# E_list=[]
# for index, tuple_stand_ACset in enumerate(loaded_dict['H1-Li1_STO-3G_singlet']):
    
#     AC_set = tuple_stand_ACset[1]
#     if index == 0:
#         E_list.append(list(AC_set[0][0].terms.values())[0]) # <- no qubits!
#     else:
#         n_qubits = qubit_ordering[index]
#         N_dict = {key:0 for key in AC_set}
#         E_LCU = LCU_linalg_Energy(AC_set,
#                                   N_dict,
#                                   n_qubits,
#                                   atol=1e-8,
#                                   rtol=1e-05,
#                                   check_reduction=False)
#         E_list.append(copy(E_LCU))
# E_LCU_dict['H1-Li1_STO-3G_singlet'] = E_list

# print(E_LCU_dict)



# mol_name = 'Ne1_STO-3G_singlet'
# AC_set= loaded_dict[mol_name][-1][1]
# N_dict = {key:0 for key in AC_set}
# n_qubits = hamiltonians[mol_name][1]

# pprint.pprint(AC_set)

# E_LCU = LCU_linalg_Energy(AC_set,
#                           N_dict,
#                           n_qubits,
#                           atol=1e-8,
#                           rtol=1e-05,
#                           check_reduction=False)

# print(E_LCU)

mol_name = 'Ne1_STO-3G_singlet'
check_reduction = True
atol=1e-8
rtol=1e-5

anti_commuting_sets= loaded_dict[mol_name][-1][1]
N_indices_dict = {key:0 for key in anti_commuting_sets}
N_Qubits = hamiltonians[mol_name][1]

reduced_H_matrix = csc_matrix((2 ** N_Qubits, 2 ** N_Qubits), dtype=complex)

H_single_terms = QubitOperator()

for key in anti_commuting_sets:
    AC_set = anti_commuting_sets[key]

    if len(AC_set) < 2:
        H_single_terms += AC_set[0]
    else:
        N_index = N_indices_dict[key]

        R_uncorrected, Pn, gamma_l = Get_R_op_list(AC_set, N_index)
        #     R_corrected_Op_list, R_corr_list, ancilla_amplitudes, l1_norm = absorb_complex_phases(R_uncorrected)

        R = QubitOperator()
        for op in R_uncorrected:
            R += op

        R_mat = qubit_operator_sparse(R, n_qubits=N_Qubits)
        Pn_mat = qubit_operator_sparse(Pn, n_qubits=N_Qubits)

        RPR_matrix = R_mat.conj().transpose().dot(
            Pn_mat.dot(R_mat))  # note this is R^{dag}PR and NOT: RHR^{dag}

        if check_reduction:
            full_normalised_set = Get_beta_j_cofactors(AC_set)
            H_S = QubitOperator()
            for QubitOp in full_normalised_set['PauliWords']:
                H_S += QubitOp
            H_S_matrix = qubit_operator_sparse(H_S, n_qubits=N_Qubits)

            RHR_matrix = R_mat.dot(H_S_matrix.dot(R_mat.conj().transpose()))
            if sparse_allclose(Pn_mat, RHR_matrix, atol=atol, rtol=rtol) is not True:
                raise ValueError('error in unitary partitioning reduction')

        reduced_H_matrix += RPR_matrix * gamma_l

reduced_H_matrix += qubit_operator_sparse(H_single_terms, n_qubits=N_Qubits)


print(reduced_H_matrix)
from scipy.linalg import eigh
a,b = eigsh(reduced_H_matrix.todense())
print(a)