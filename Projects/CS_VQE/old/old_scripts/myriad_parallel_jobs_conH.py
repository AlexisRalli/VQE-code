# NOT CORRECT!


# import numpy as np
# import cs_vqe as c
# import ast
# import os
# from tqdm import tqdm
# import scipy as sp

# import cs_vqe_with_LCU as c_LCU

# import pickle
# import datetime

# #######
# import sys
# # working_dir = os.getcwd()
# working_dir = os.path.dirname(os.path.abspath(__file__)) # gets directory where running python file is!
# data_dir = os.path.join(working_dir, 'data')
# data_hamiltonians_file = os.path.join(data_dir, 'hamiltonians.txt')

# data_csvqe_results_file = os.path.join(data_dir, 'csvqe_results.txt')
# with open(data_csvqe_results_file, 'r') as input_file:
#     csvqe_results = ast.literal_eval(input_file.read())


# print('start time: {}'.format(datetime.datetime.now().strftime('%Y%b%d-%H%M%S%f')))
# print('working directory:', working_dir)

# with open(data_hamiltonians_file, 'r') as input_file:
#     hamiltonians = ast.literal_eval(input_file.read())

# for key in csvqe_results.keys():
#     print(f"{key: <25}     n_qubits:  {hamiltonians[key][1]:<5.0f}")



# ### run calc ##

# N_index=0
# check_reduction=False

# csvqe_LCU_output={}
# csvqe_standard_output={}

# ######## take commandline arguement to run in parallel
# mol_num = int(sys.argv[1]) 

# molecules_and_qubits = [(mol_key, hamiltonians[mol_key][1])for mol_key in csvqe_results]
# sorted_by_qubit_No = sorted(molecules_and_qubits, key= lambda x: x[1])
# mol_key = sorted_by_qubit_No[mol_num-1][0] # UCL supercomputer indexes from 1, hence minus one here!
# ########

# true_gs = hamiltonians[mol_key][4] # ground state energy of full Hamiltonian (in Hartree)
# ham_noncon = hamiltonians[mol_key][3]  # noncontextual part of Hamiltonian, found by greedy DFS
# ham = hamiltonians[mol_key][2] # full Hamiltonian
# gs_noncon = hamiltonians[mol_key][5]
# model = list(gs_noncon[3])
# fn_form = gs_noncon[4]
# N_Qubits= hamiltonians[mol_key][1]
# ground_state_params = [list(gs_noncon[1]), list(gs_noncon[2])]


# ham_CON = {}
# for P in ham:
#     if P in ham_noncon:
#         continue
#     else:
#         ham_CON[P]=ham[P]


# removal_order = csvqe_results[mol_key][3]


# ### LCU method!
# reduced_H_LCU_list = c_LCU.get_reduced_hamiltonians_LCU(ham_CON,
#                          model,
#                          fn_form,
#                          ground_state_params,
#                          deepcopy(removal_order), 
#                          N_Qubits,
#                          N_index, 
#                          check_reduction=check_reduction)


# ### Seq Rot method (standard)
# reduced_H_standard_list = c.get_reduced_hamiltonians(ham_CON, #<-- CON PART ONLY
#                      model,
#                      fn_form,
#                      ground_state_params,
#                      deepcopy(removal_order))


# Energy_function = c.energy_function(fn_form)
# non_con_GS = Energy_function(*gs_noncon[1], *gs_noncon[2]) #<-- needs to be added to results!



# ### Find contextual_energies
# H_std_conH_results={}
# H_std_conH_results['NonCon'] = {'E_noncon': non_con_GS, 'H_noncon':ham_noncon}
# H_std_conH_results['Con'] = {}
# for ind, H_std in enumerate(reduced_H_standard_list):
#     if H_std:
#         ham_red_sparse = c.hamiltonian_to_sparse(H_std)
#         if ham_red_sparse.shape[0] <= 64:
#             E_con = min(np.linalg.eigvalsh(ham_red_sparse.toarray()))
#         else:
#           eig_values, eig_vectors = eigsh(ham_red_sparse, k=1, which='SA')
#           E_con = min(eig_values)
#     else:
#         # case for full approximation (when H_std = {})
#         E_con=0
    
#     H_std_conH_results['Con'][ind] = {'E_con': E_con, 'H_con':H_std}


# H_LCU_conH_results={}
# H_LCU_conH_results['NonCon'] = {'E_noncon': non_con_GS, 'H_noncon':ham_noncon}
# H_LCU_conH_results['Con'] = {}
# for ind, H_LCU in enumerate(reduced_H_LCU_list):
#     if H_LCU:
#         ham_red_sparse = c.hamiltonian_to_sparse(H_LCU)
#         if ham_red_sparse.shape[0] <= 64:
#             E_con = min(np.linalg.eigvalsh(ham_red_sparse.toarray()))
#         else:
#           eig_values, eig_vectors = eigsh(ham_red_sparse, k=1, which='SA')
#           E_con = min(eig_values)
#     else:
#         # case for full approximation (when H_std = {})
#         E_con=0
    
#     H_LCU_conH_results['Con'][ind] = {'E_con': E_con, 'H_con':H_LCU}



# ####### SAVE OUTPUT details
# unique_file_time = datetime.datetime.now().strftime('%Y%b%d-%H%M%S%f')
# # output_dir = os.path.join(working_dir, 'Pickle_out')
# output_dir = os.getcwd()
# ########


# ####### SAVE OUTPUT
# file_name1 = 'REDUCED_H_con_standard_CS_VQE_exp__{}__{}_.pickle'.format(unique_file_time, mol_key)
# file_out1=os.path.join(output_dir, file_name1)
# with open(file_out1, 'wb') as outfile:
#     pickle.dump(H_std_conH_results, outfile)


# file_name2 = 'REDUCED_H_con_LCU_CS_VQE_exp__{}__{}_.pickle'.format(unique_file_time, mol_key)
# file_out2=os.path.join(output_dir, file_name2)
# with open(file_out2, 'wb') as outfile:
#     pickle.dump(H_LCU_conH_results, outfile)


# print('pickle files dumped unqiue time id: {}'.format(unique_file_time))

# print('end time: {}'.format(datetime.datetime.now().strftime('%Y%b%d-%H%M%S%f')))