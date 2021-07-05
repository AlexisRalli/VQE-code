import numpy as np
import scipy as sp

import ast
import os

from quchem.Unitary_Partitioning.Graph import Clique_cover_Hamiltonian
import quchem.Misc_functions.conversion_scripts as conv_scr 
from copy import deepcopy
from quchem.Unitary_Partitioning.Unitary_partitioning_Seq_Rot import SeqRot_linalg_Energy

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



### find anti-commuting sets
unitary_paritioning_SeqRot={}

# optional params!
commutativity_flag = 'AC' ## <- defines relationship between sets!!!
Graph_colouring_strategy='largest_first'
check_reduction_SeqRot = False
prune_threshold = 1e-6

######## take commandline arguement to run in parallel
# sys.argv[0] = python file_name
mol_key_index = int(sys.argv[1])-1 # minus one as array script idexes from 1
mol_key = sorted(list(myriad_SeqRot_results.keys()))[mol_key_index]
if mol_key not in myriad_SeqRot_results.keys():
    raise ValueError('molecule key not correct')

########



####### SAVE OUTPUT details
# output_dir = os.path.join(working_dir, 'Pickle_out')

AC_sets_dir_name = ' AC_sets_SeqRot'
AC_dir = os.path.join(working_dir, AC_sets_dir_name) # saves in VQE-code area! (not Scratch)
# Create target Directory if it doesn't exist
if not os.path.exists(AC_dir):
    os.mkdir(AC_dir)




# for mol_key in tqdm(list(myriad_LCU_results.keys())): # removed loop and used myriad array input!
anti_commuting_sets_different_H_LCU_sizes={}
for ind_key in myriad_SeqRot_results[mol_key]:
    
    if isinstance(ind_key, str):
        continue
    
    if ind_key==0:
        # only non-contextual problem
        anti_commuting_sets_different_H_LCU_sizes[ind_key]= {'AC_sets':{}, 'ground_state': None}
    else:
        ### SeqRot
        H_SeqRot_dict = myriad_SeqRot_results[mol_key][ind_key]['H']
        H_SeqRot_pruned = {P_key: coeff.real for P_key, coeff in H_SeqRot_dict.items() if np.abs(coeff)>prune_threshold}

        H_SeqRot= conv_scr.Get_Openfermion_Hamiltonian(H_SeqRot_pruned)
        n_qubits = len(list(H_SeqRot_dict.keys())[0])
        
        anti_commuting_sets_SeqRot = Clique_cover_Hamiltonian(H_SeqRot, 
                                         n_qubits, 
                                         commutativity_flag, 
                                         Graph_colouring_strategy)

        reduced_H_matrix = qubit_operator_sparse(H_SeqRot, n_qubits=n_qubits)
        if reduced_H_matrix.shape[0]<=64:
            eig_values, eig_vectors = eigh(reduced_H_matrix.todense()) # NOT sparse!
        else:
            eig_values, eig_vectors = eigsh(reduced_H_matrix, k=1, which='SA') # < solves eigenvalue problem for a complex Hermitian matrix.


        idx = eig_values.argsort()[::-1]   
        eigenValues = eig_values[idx]
        eigenVectors = eig_vectors[:,idx]

        anti_commuting_sets_different_H_LCU_sizes[ind_key]= {'AC_sets':anti_commuting_sets_SeqRot, 'ground_state': eigenVectors[:,0]} 


# save file
file_out1 = os.path.join(AC_dir, mol_key)

####### SAVE OUTPUT
with open(file_out1, 'wb') as outfile:
    pickle.dump(anti_commuting_sets_different_H_LCU_sizes, outfile)


print('pickle files dumped at: {}'.format(file_out1))