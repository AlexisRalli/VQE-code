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

ind_key  = int(sys.argv[1])-1 # minus one as array script idexes from 1
mol_key = sys.argv[2]

 if mol_key not in myriad_SeqRot_results.keys():
  raise ValueError('molecule key not correct')

########


anti_commuting_sets_different_H_SeqRot_sizes={}
if ind_key==0:
    # only non-contextual problem
    anti_commuting_sets_different_H_SeqRot_sizes[ind_key]= {'AC_sets': {},
                                                           'E':myriad_SeqRot_results[mol_key][ind_key]['E']}
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

    all_zero_Ps_index_dict = {set_key: 0 for set_key in anti_commuting_sets_SeqRot}
    E_SeqRot = SeqRot_linalg_Energy(anti_commuting_sets_SeqRot,
                                     all_zero_Ps_index_dict,
                                     n_qubits,
                                     atol=1e-8,
                                     rtol=1e-05,
                                     check_reduction=check_reduction_SeqRot)

    anti_commuting_sets_different_H_SeqRot_sizes[ind_key]= {'AC_sets': anti_commuting_sets_SeqRot,
                                                           'E':E_SeqRot}


unitary_paritioning_SeqRot[mol_key]= deepcopy(anti_commuting_sets_different_H_SeqRot_sizes)
del anti_commuting_sets_different_H_SeqRot_sizes


####### SAVE OUTPUT details
unique_file_time = datetime.datetime.now().strftime('%Y%b%d-%H%M%S%f')
# output_dir = os.path.join(working_dir, 'Pickle_out')
working_directory = os.getcwd()
output_dir =os.path.join(working_directory, mol_key)

# Create target Directory if it doesn't exist
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


# save file
file_name1 = 'Unitary_Partitinging_SeqRot_CS_VQE_SeqRot_exp__{}__{}__{}.pickle'.format(unique_file_time, mol_key, ind_key)
file_out1=os.path.join(output_dir, file_name1)

####### SAVE OUTPUT
with open(file_out1, 'wb') as outfile:
    pickle.dump(unitary_paritioning_SeqRot, outfile)


print('pickle files dumped unqiue time id: {}'.format(unique_file_time))

print('end time: {}'.format(datetime.datetime.now().strftime('%Y%b%d-%H%M%S%f')))