import numpy as np
import scipy as sp

import ast
import os
from copy import deepcopy

import cs_vqe_updated_up as c_new
import quchem.Misc_functions.conversion_scripts as conv_scr
from openfermion import qubit_operator_sparse
from quchem.Unitary_Partitioning.Graph import Clique_cover_Hamiltonian
from quchem.Unitary_Partitioning.Unitary_partitioning_LCU_method import LCU_linalg_Energy
from quchem.Unitary_Partitioning.Unitary_partitioning_Seq_Rot import SeqRot_linalg_Energy

import pickle
import datetime

#######
import sys
# working_dir = os.getcwd()

#### optional params!
check_reduction=False
commutativity_flag = 'AC' ## <- defines relationship between sets!!!
Graph_colouring_strategy='largest_first'
check_reduction_LCU = False
prune_threshold = 1e-9
####


working_dir = os.path.dirname(os.path.abspath(__file__)) # gets directory where running python file is!
data_dir = os.path.join(working_dir, 'data')
data_hamiltonians_file = os.path.join(data_dir, 'hamiltonians.txt')

print('start time: {}'.format(datetime.datetime.now().strftime('%Y%b%d-%H%M%S%f')))
print('working directory:', working_dir)

with open(data_hamiltonians_file, 'r') as input_file:
    hamiltonians = ast.literal_eval(input_file.read())

data_csvqe_results_file = os.path.join(data_dir, 'csvqe_results.txt')
with open(data_csvqe_results_file, 'r') as input_file:
    csvqe_results = ast.literal_eval(input_file.read())

for key in csvqe_results.keys():
    print(f"{key: <25}     n_qubits:  {hamiltonians[key][1]:<5.0f}")


######## take commandline arguement to run in parallel
mol_num = int(sys.argv[1]) 

molecules_and_qubits = [(mol_key, hamiltonians[mol_key][1])for mol_key in csvqe_results]
sorted_by_qubit_No = sorted(molecules_and_qubits, key= lambda x: x[1])
mol_key = sorted_by_qubit_No[mol_num-1][0] # UCL supercomputer indexes from 1, hence minus one here!

########

true_gs = hamiltonians[mol_key][4] # ground state energy of full Hamiltonian (in Hartree)
ham_noncon = hamiltonians[mol_key][3]  # noncontextual part of Hamiltonian, found by greedy DFS
ham = hamiltonians[mol_key][2] # full Hamiltonian
gs_noncon = hamiltonians[mol_key][5]
model = list(gs_noncon[3])
fn_form = gs_noncon[4]
ground_state_params = [list(gs_noncon[1]), list(gs_noncon[2])]
removal_order = csvqe_results[mol_key][3]

exp_conditions = {
    'true_gs':true_gs,
    'noncon_H':ham_noncon,
    'full_tapered_H': ham,
    'gstate_noncon': [list(gs_noncon[1]), list(gs_noncon[2])],
    'gstate_noncon_Energy': gs_noncon[0],
    'model_CSVQE': list(gs_noncon[3]),
    'fn_form_CSVQE': gs_noncon[4],
    'n_qubits':   hamiltonians[mol_key][1],
    'removal_order': removal_order,
    'mol_key': mol_key,
    'old_energy_diff': csvqe_results[mol_key][2]
}

###

### LCU method (no M reduction)!
reduced_H_csvqe_LCU = c_new.get_reduced_H_minimal_rotations(ham,
                         model,
                         fn_form,
                         ground_state_params,
                         deepcopy(removal_order), 
                         check_reduction=check_reduction,
                         up_method='LCU')

### Seq Rot method (no M reduction)!
reduced_H_csvqe_seqrot = c_new.get_reduced_H_minimal_rotations(ham,
                         model,
                         fn_form,
                         ground_state_params,
                         deepcopy(removal_order),
                         check_reduction=check_reduction,
                         up_method='SeqRot')

### Get cs_vqe_seqrot no M reduction energies
SeqRot_results={}
for ind, H_seqrot in enumerate(reduced_H_csvqe_seqrot['H_csvqe']):
    Ham_openF = conv_scr.Get_Openfermion_Hamiltonian(H_seqrot)
    ham_red_sparse = qubit_operator_sparse(Ham_openF)
    if ham_red_sparse.shape[0] <= 128:
        Energy = min(np.linalg.eigvalsh(ham_red_sparse.toarray()))
    else:
        eig_values, eig_vectors = sp.sparse.linalg.eigsh(ham_red_sparse, k=1, which='SA')
        Energy = min(eig_values)

    del Ham_openF
    del ham_red_sparse
    SeqRot_results[ind] = {'E': Energy , 'H': H_seqrot}
    

### Get cs_vqe_LCU no M reduction energies
LCU_results={}
for ind, H_LCU in enumerate(reduced_H_csvqe_LCU['H_csvqe']):
    Ham_openF = conv_scr.Get_Openfermion_Hamiltonian(H_LCU)
    ham_red_sparse = qubit_operator_sparse(Ham_openF)
    if ham_red_sparse.shape[0] <= 128:
        Energy = min(np.linalg.eigvalsh(ham_red_sparse.toarray()))
    else:
        eig_values, eig_vectors = sp.sparse.linalg.eigsh(ham_red_sparse, k=1, which='SA')
        Energy = min(eig_values)

    del Ham_openF
    del ham_red_sparse
    LCU_results[ind] = {'E': Energy , 'H':H_LCU}

### with measurement reduction
m_red_seqrot_with_seqrot_cs_vqe = {}
for ind_key, H_seqrot in enumerate(reduced_H_csvqe_seqrot['H_csvqe']):
    if ind_key == 0:
        # only non-contextual problem
        m_red_seqrot_with_seqrot_cs_vqe[ind_key] = {'AC_sets': {}, 'energy': 0}
    else:
        ### SeqRot
        H_SeqRot_pruned = {P_key: coeff.real for P_key, coeff in H_seqrot.items() if
                           np.abs(coeff) > prune_threshold}

        H_SeqRot = conv_scr.Get_Openfermion_Hamiltonian(H_SeqRot_pruned)
        n_qubits = len(list(H_seqrot.keys())[0])

        anti_commuting_sets_SeqRot = Clique_cover_Hamiltonian(H_SeqRot,
                                                              n_qubits,
                                                              commutativity_flag,
                                                              Graph_colouring_strategy)

        # setting S_indices to zero for all terms!
        S_key_dict = dict((zip(list(anti_commuting_sets_SeqRot.keys()),
                               [0 for _ in range(len(anti_commuting_sets_SeqRot.keys()))])))

        fci_energy = SeqRot_linalg_Energy(anti_commuting_sets_SeqRot,
                                          S_key_dict,
                                          n_qubits,
                                          atol=1e-8,
                                          rtol=1e-05,
                                          check_reduction=check_reduction)

        m_red_seqrot_with_seqrot_cs_vqe[ind_key] = {'AC_sets': anti_commuting_sets_SeqRot,
                                                              'energy': fci_energy}


    ####### SAVE OUTPUT details

m_red_LCU_with_LCU_cs_vqe = {}
for ind_key, H_LCU in enumerate(reduced_H_csvqe_LCU['H_csvqe']):
    if ind_key == 0:
        # only non-contextual problem
        m_red_LCU_with_LCU_cs_vqe[ind_key] = {'AC_sets': {}, 'energy': 0}
    else:
        ### LCU
        H_LCU_pruned = {P_key: coeff.real for P_key, coeff in H_LCU.items() if
                           np.abs(coeff) > prune_threshold}

        H_LCU_openf = conv_scr.Get_Openfermion_Hamiltonian(H_LCU_pruned)
        n_qubits = len(list(H_LCU_pruned.keys())[0])

        anti_commuting_sets_LCU = Clique_cover_Hamiltonian(H_LCU_openf,
                                                              n_qubits,
                                                              commutativity_flag,
                                                              Graph_colouring_strategy)

        # setting S_indices to zero for all terms!
        S_key_dict = dict((zip(list(anti_commuting_sets_LCU.keys()),
                               [0 for _ in range(len(anti_commuting_sets_LCU.keys()))])))

        fci_energy = LCU_linalg_Energy(anti_commuting_sets_LCU,
                                          S_key_dict,
                                          n_qubits,
                                          atol=1e-8,
                                          rtol=1e-05,
                                          check_reduction=check_reduction)

        m_red_LCU_with_LCU_cs_vqe[ind_key] = {'AC_sets': anti_commuting_sets_LCU,
                                                    'energy': fci_energy}


####### SAVE OUTPUT details
unique_file_time = datetime.datetime.now().strftime('%Y%b%d-%H%M%S%f')
# output_dir = os.path.join(working_dir, 'Pickle_out')
output_dir = os.getcwd()
########

output_data = {}
output_data['exp_conditions'] = exp_conditions


output_data['cs_vqe_seqrot'] = reduced_H_csvqe_seqrot
output_data['cs_vqe_LCU'] = reduced_H_csvqe_LCU

# without measurement reduction
output_data['cs_vqe_seqrot_energies'] = SeqRot_results
output_data['cs_vqe_LCU_energies'] = LCU_results

# with measurement reduction
output_data['cs_vqe_seqrot_M_seqrot'] = m_red_seqrot_with_seqrot_cs_vqe
output_data['cs_vqe_LCU_M_LCU'] = m_red_LCU_with_LCU_cs_vqe


####### SAVE OUTPUT
file_name1 = 'exp_results__{}__{}_.pickle'.format(unique_file_time, mol_key)
file_out1=os.path.join(output_dir, file_name1)
with open(file_out1, 'wb') as outfile:
    pickle.dump(output_data, outfile)

print('pickle files dumped unqiue time id: {}'.format(unique_file_time))
print('end time: {}'.format(datetime.datetime.now().strftime('%Y%b%d-%H%M%S%f')))

