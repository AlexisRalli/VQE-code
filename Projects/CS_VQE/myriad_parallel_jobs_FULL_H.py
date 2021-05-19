import numpy as np
import cs_vqe as c
import ast
import os
from tqdm import tqdm
from copy import deepcopy

import cs_vqe_with_LCU as c_LCU
import quchem.Misc_functions.conversion_scripts as conv_scr 
from openfermion import qubit_operator_sparse

import pickle
import datetime

#######
import sys
# working_dir = os.getcwd()
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



### run calc ##
check_reduction=False

csvqe_LCU_output={}
csvqe_standard_output={}

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
    'fn_form_CSVQE':gs_noncon[4],
    'n_qubits':   hamiltonians[mol_key][1],
    'removal_order': removal_order,
    'mol_key': mol_key
}

###
### LCU method!
reduced_H_LCU_list = c_LCU.get_reduced_hamiltonians_LCU(ham,
                         model,
                         fn_form,
                         ground_state_params,
                         deepcopy(removal_order), 
                         check_reduction=check_reduction)


### Seq Rot method (standard)
reduced_H_SeqRot_list = c.get_reduced_hamiltonians(ham, #<-- CON PART ONLY
                     model,
                     fn_form,
                     ground_state_params,
                     deepcopy(removal_order))



### Get SeqRot Energies
SeqRot_results={}
for ind, H_std in enumerate(reduced_H_SeqRot_list):
    Ham_openF = conv_scr.Get_Openfermion_Hamiltonian(H_std)
    ham_red_sparse = qubit_operator_sparse(Ham_openF)
    if ham_red_sparse.shape[0] <= 64:
        Energy = min(np.linalg.eigvalsh(ham_red_sparse.toarray()))
    else:
        eig_values, eig_vectors = eigsh(ham_red_sparse, k=1, which='SA')
        Energy = min(eig_values)
    SeqRot_results[ind] = {'E':Energy , 'H':H_std}
    
SeqRot_results['exp_conditions'] = exp_conditions


### Get LCU Energies
LCU_results={}
for ind, H_LCU in enumerate(reduced_H_LCU_list):
    Ham_openF = conv_scr.Get_Openfermion_Hamiltonian(H_LCU)
    ham_red_sparse = qubit_operator_sparse(Ham_openF)
    if ham_red_sparse.shape[0] <= 64:
        Energy = min(np.linalg.eigvalsh(ham_red_sparse.toarray()))
    else:
        eig_values, eig_vectors = eigsh(ham_red_sparse, k=1, which='SA')
        Energy = min(eig_values)
    LCU_results[ind] = {'E':Energy , 'H':H_LCU}
    
LCU_results['exp_conditions'] = exp_conditions



####### SAVE OUTPUT details
unique_file_time = datetime.datetime.now().strftime('%Y%b%d-%H%M%S%f')
# output_dir = os.path.join(working_dir, 'Pickle_out')
output_dir = os.getcwd()
########


####### SAVE OUTPUT
file_name1 = 'SeqRot_CS_VQE_exp__{}__{}_.pickle'.format(unique_file_time, mol_key)
file_out1=os.path.join(output_dir, file_name1)
with open(file_out1, 'wb') as outfile:
    pickle.dump(SeqRot_results, outfile)


file_name2 = 'LCU_CS_VQE_exp__{}__{}_.pickle'.format(unique_file_time, mol_key)
file_out2=os.path.join(output_dir, file_name2)
with open(file_out2, 'wb') as outfile:
    pickle.dump(LCU_results, outfile)


print('pickle files dumped unqiue time id: {}'.format(unique_file_time))

print('end time: {}'.format(datetime.datetime.now().strftime('%Y%b%d-%H%M%S%f')))