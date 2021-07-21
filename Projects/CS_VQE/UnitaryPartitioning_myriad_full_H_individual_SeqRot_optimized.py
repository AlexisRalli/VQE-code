import numpy as np
import scipy as sp

import ast
import os
import re

from scipy.sparse import csc_matrix
from quchem.Unitary_Partitioning.Graph import Clique_cover_Hamiltonian
import quchem.Misc_functions.conversion_scripts as conv_scr 
from copy import deepcopy
from quchem.Unitary_Partitioning.Unitary_partitioning_Seq_Rot import Get_reduced_H_matrix_SeqRot_matrix_FAST, Get_reduced_H_matrix_SeqRot

from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

from openfermion import qubit_operator_sparse

import pickle
import datetime

#######
import sys
# working_dir = os.getcwd()
working_dir = os.path.dirname(os.path.abspath(__file__)) # gets directory where running python file is!
Analysis_dir = os.path.join(working_dir, 'Analysis')
full_H_results_dir = os.path.join(Analysis_dir, 'SeqRot_LCU_script_A_results')


print('start time: {}'.format(datetime.datetime.now().strftime('%Y%b%d-%H%M%S%f')))
print('working directory:', working_dir)



###### IMPORT INITIAL RESULTS


## import SeqRot results
myriad_SeqRot_results = {}
for filename in os.listdir(full_H_results_dir):
    if (filename.endswith('.pickle') and filename.startswith('SeqRot_CS_VQE_exp')):
        file_path = os.path.join(full_H_results_dir, filename) 
        mol_name = filename[43:-8]
        with open(file_path,'rb') as infile:
            data = pickle.load(infile)
        myriad_SeqRot_results[mol_name] = data





######## take commandline arguement to run in parallel
# sys.argv[0] = python file_name
AC_set_index  = int(sys.argv[1])-1 # minus one as array script idexes from 1
mol_key = sys.argv[2]
decimal_place_threshold = 10 # in ground state ket remove terms with vals lower than 1e-10
check_reduction_SeqRot = False

if mol_key not in myriad_SeqRot_results.keys():
    raise ValueError('molecule key not correct')



########
## import AC_sets results

AC_sets_dir_name = 'AC_sets_SeqRot'
AC_dir = os.path.join(working_dir, AC_sets_dir_name)
input_AC_file_path = os.path.join(AC_dir, mol_key + '.pickle') # AC of given molecule


with open(input_AC_file_path,'rb') as infile:
    all_anti_commuting_sets_SeqRot = pickle.load(infile)

anti_commuting_sets_SeqRot = all_anti_commuting_sets_SeqRot[AC_set_index]['AC_sets']
ground_state_ket = all_anti_commuting_sets_SeqRot[AC_set_index]['ground_state']


## Get Energy

if anti_commuting_sets_SeqRot:
    ### SeqRot
    all_zero_Ps_index_dict = {set_key: 0 for set_key in anti_commuting_sets_SeqRot}

    H_SeqRot_dict = myriad_SeqRot_results[mol_key][AC_set_index]['H']
    n_qubits = len(list(H_SeqRot_dict.keys())[0])

    H_sparse = Get_reduced_H_matrix_SeqRot(anti_commuting_sets_SeqRot,
                                     all_zero_Ps_index_dict,
                                     n_qubits,
                                     atol=1e-8,
                                     rtol=1e-05,
                                     check_reduction=check_reduction_SeqRot)

    # denisty_mat = np.outer(ground_state_ket, ground_state_ket)
    # E_SeqRot = np.trace(denisty_mat@H_sparse)

    # if n_qubits<6:
    #     eig_values, eig_vectors = eigh(H_sparse.todense()) # NOT sparse!
    # else:
    #     eig_values, eig_vectors = eigsh(H_sparse, k=1, which='SA') # < solves eigenvalue problem for a complex Hermitian matrix.

    # E_SeqRot = min(eig_values)
    # AC_set_and_Energy_output = {'AC_sets': anti_commuting_sets_SeqRot,
    #                                                        'E':E_SeqRot}

    sparse_ket = csc_matrix(np.around(ground_state_ket,decimal_place_threshold).reshape([ground_state_ket.shape[0],1]), dtype=complex)

    E_SeqRot = sparse_ket.conj().T @ H_sparse @ sparse_ket

    AC_set_and_Energy_output = {'AC_sets': anti_commuting_sets_SeqRot,
                                                   'E':E_SeqRot.todense().item(0)}
else:
    # only non-contextual problem
    AC_set_and_Energy_output = {'AC_sets': anti_commuting_sets_SeqRot,
                                                           'E':myriad_SeqRot_results[mol_key][AC_set_index]['E']}    


####### SAVE OUTPUT details
unique_file_time = datetime.datetime.now().strftime('%Y%b%d-%H%M%S%f')
working_directory = os.getcwd()
output_dir =os.path.join(working_directory, mol_key)

# Create target Directory if it doesn't exist
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


# save file
file_name1 = 'AC_set_and_Energy_output_set_key_{}.pickle'.format(AC_set_index)
file_out1=os.path.join(output_dir, file_name1)

####### SAVE OUTPUT
with open(file_out1, 'wb') as outfile:
    pickle.dump(AC_set_and_Energy_output, outfile)


print('pickle files dumped at: {}'.format(file_out1))

print('end time: {}'.format(datetime.datetime.now().strftime('%Y%b%d-%H%M%S%f')))