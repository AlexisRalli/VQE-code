import numpy as np
import scipy as sp

import ast
import os

from quchem.Unitary_Partitioning.Graph import Clique_cover_Hamiltonian
import quchem.Misc_functions.conversion_scripts as conv_scr 
from copy import deepcopy
from quchem.Unitary_Partitioning.Unitary_partitioning_LCU_method import LCU_linalg_Energy

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

## import LCU results
myriad_LCU_results = {}
for filename in os.listdir(full_H_results_dir):
    if (filename.endswith('.pickle') and filename.startswith('LCU_CS_VQE_exp')):
        file_path = os.path.join(full_H_results_dir, filename) 
        mol_name = filename[40:-8]
        with open(file_path,'rb') as infile:
            data = pickle.load(infile)

        myriad_LCU_results[mol_name] = data



### find anti-commuting sets
unitary_paritioning_LCU={}

# optional params!
commutativity_flag = 'AC' ## <- defines relationship between sets!!!
Graph_colouring_strategy='largest_first'
check_reduction_LCU = False


######## take commandline arguement to run in parallel
mol_num = int(sys.argv[1]) 
sorted_mol_names = sorted(list(myriad_LCU_results.keys()))
mol_key = sorted_mol_names[mol_num-1] # UCL supercomputer indexes from 1, hence minus one here!
########


# for mol_key in tqdm(list(myriad_LCU_results.keys())): # removed loop and used myriad array input!
anti_commuting_sets_different_H_LCU_sizes={}
for ind_key in myriad_LCU_results[mol_key]:
    
    if isinstance(ind_key, str):
        continue
    
    if ind_key==0:
        # only non-contextual problem
        anti_commuting_sets_different_H_LCU_sizes[ind_key]= {'AC_sets': {},
                                                             'E':myriad_LCU_results[mol_key][ind_key]['E']}
    else:

        ### LCU
        H_LCU_dict = myriad_LCU_results[mol_key][ind_key]['H']
        H_LCU_pruned = {P_key: coeff.real for P_key, coeff in H_LCU_dict.items() if not np.isclose(coeff.real,0)}
        
        H_LCU= conv_scr.Get_Openfermion_Hamiltonian(H_LCU_pruned)

        n_qubits = len(list(H_LCU_dict.keys())[0])
        anti_commuting_sets_LCU = Clique_cover_Hamiltonian(list(H_LCU), 
                                                     n_qubits, 
                                                     commutativity_flag, 
                                                     Graph_colouring_strategy)
        

        all_zero_Pn_index_dict = {set_key: 0 for set_key in anti_commuting_sets_LCU}
        E_LCU = LCU_linalg_Energy(anti_commuting_sets_LCU,
              all_zero_Pn_index_dict, 
              n_qubits, 
              atol=1e-8, 
              rtol=1e-05,
              check_reduction=check_reduction_LCU)

        anti_commuting_sets_different_H_LCU_sizes[ind_key]= {'AC_sets': anti_commuting_sets_LCU,
                                                             'E':E_LCU}


unitary_paritioning_LCU[mol_key]= deepcopy(anti_commuting_sets_different_H_LCU_sizes)
del anti_commuting_sets_different_H_LCU_sizes


####### SAVE OUTPUT details
unique_file_time = datetime.datetime.now().strftime('%Y%b%d-%H%M%S%f')
# output_dir = os.path.join(working_dir, 'Pickle_out')
output_dir = os.getcwd()
########


####### SAVE OUTPUT
file_name2 = 'Unitary_Partitinging_LCU_CS_VQE_LCU_exp__{}__{}_.pickle'.format(unique_file_time, mol_key)
file_out2=os.path.join(output_dir, file_name2)

with open(file_out2, 'wb') as outfile:
    pickle.dump(unitary_paritioning_LCU, outfile)


print('pickle files dumped unqiue time id: {}'.format(unique_file_time))

print('end time: {}'.format(datetime.datetime.now().strftime('%Y%b%d-%H%M%S%f')))