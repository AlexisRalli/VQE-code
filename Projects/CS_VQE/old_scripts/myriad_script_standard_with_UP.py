import numpy as np
import cs_vqe as c
from copy import deepcopy as copy
from tqdm import tqdm
import pickle
import datetime

import quchem.Misc_functions.conversion_scripts as conv_scr
from quchem.Unitary_Partitioning.Graph import Clique_cover_Hamiltonian
from quchem.Unitary_Partitioning.Unitary_partitioning_Seq_Rot import SeqRot_linalg_Energy
from quchem.Unitary_Partitioning.Unitary_partitioning_LCU_method import LCU_linalg_Energy


import ast
import matplotlib.pyplot as plt
import os


# working_dir = os.getcwd()
working_dir = os.path.dirname(os.path.abspath(__file__)) # gets directory where running python file is!
data_dir = os.path.join(working_dir, 'data')
data_hamiltonians_file = os.path.join(data_dir, 'hamiltonians.txt')

print('start time: {}'.format(datetime.datetime.now().strftime('%Y%b%d-%H%M%S%f')))
print('working directory:', working_dir)

with open(data_hamiltonians_file, 'r') as input_file:
    hamiltonians = ast.literal_eval(input_file.read())

for key in hamiltonians.keys():
    print(f"{key: <25}     n_qubits:  {hamiltonians[key][1]:<5.0f}")


## import data
data_csvqe_results_file = os.path.join(data_dir, 'csvqe_results.txt')
with open(data_csvqe_results_file, 'r') as input_file:
    csvqe_results = ast.literal_eval(input_file.read())


updated_hamiltonians={}
for key in csvqe_results.keys(): # <- note using csvqe_results (has less molecules than hamiltonians.keys())
    

    ham_noncon = hamiltonians[key][3]
    ep_state = [list(hamiltonians[key][5][1]), list(hamiltonians[key][5][2])] # ground state of noncontextual Ham
    ham = hamiltonians[key][2]

    model = c.quasi_model(ham_noncon)
    fn_form = c.energy_function_form(ham_noncon, model)
    
#     _, _, _, ham_out, ham_noncon_out = c.diagonalize_G(model,fn_form,ep_state, ham, ham_noncon)

    N_qubits = hamiltonians[key][1]
    order= csvqe_results[key][-1] # <-- order specified from greedy run by Will
    qubit_removal_order=copy(order)
    
    reduced_Con_hamiltonians = c.get_reduced_hamiltonians(ham ,model, fn_form, ep_state, order)
    
    
    Contextual_Hamiltonian_list=[]
    qNo_list=[]
    for qNo, Ham in enumerate(reduced_Con_hamiltonians):            
#         if c.contextualQ_ham(Ham) is False:
#             raise ValueError('have not found contextual H')
        Ham_Openf = conv_scr.Get_Openfermion_Hamiltonian(Ham)
        Contextual_Hamiltonian_list.append(list(Ham_Openf))
        qNo_list.append(qNo)
    
    updated_hamiltonians[key] = {
                                'encoding':hamiltonians[key][0],
                                'n_qubits': hamiltonians[key][1],
                                'full_H': conv_scr.Get_Openfermion_Hamiltonian(hamiltonians[key][2]),
#                                 'nonC_H_greedyDFS':Get_Openfermion_Hamiltonian(hamiltonians[key][3]),
                                'FCI':hamiltonians[key][4],                         
                                'gstate_noncon':hamiltonians[key][5], 
                                'Contextual_Hamiltonian_list': Contextual_Hamiltonian_list, #ham_out,   
                                'Contextual_Hamiltonian_qubitNo_list': qNo_list,  
                                'non_Contextual_H': conv_scr.Get_Openfermion_Hamiltonian(ham_noncon),# ham_noncon_out, 
                                'qubit_removal_order':qubit_removal_order
                             }


# checking keys match!
for key in updated_hamiltonians.keys():
    if key not in csvqe_results.keys():
        raise ValueError(f'{key} not in csvqe_results')



### find anti-commuting sets
unitary_paritioning_of_Con_H={}

commutativity_flag = 'AC' ## <- defines relationship between sets!!!
plot_graph = False
Graph_colouring_strategy='largest_first'


for key in tqdm(list(updated_hamiltonians.keys())):
    
    H_con_List = updated_hamiltonians[key]['Contextual_Hamiltonian_list']
    
    
    anti_commuting_different_red_list=[]
    for i, H_con in enumerate(H_con_List):
        n_qubits = updated_hamiltonians[key]['Contextual_Hamiltonian_qubitNo_list'][i]
        anti_commuting_sets = Clique_cover_Hamiltonian(H_con, 
                                                             n_qubits, 
                                                             commutativity_flag, 
                                                             Graph_colouring_strategy)
        
        anti_commuting_different_red_list.append(anti_commuting_sets)
    
    H_con_reduced_and_antiC_set = list(zip(H_con_List, anti_commuting_different_red_list))
    
    unitary_paritioning_of_Con_H[key] = H_con_reduced_and_antiC_set



####### SAVE OUTPUT
unique_file_time = datetime.datetime.now().strftime('%Y%b%d-%H%M%S%f')
output_dir = os.path.join(working_dir, 'Pickle_out')



file_name1 = 'experimental_ordering_anticommuting_vs_standard_conH__{}.pickle'.format(unique_file_time)
file_out1=os.path.join(output_dir, file_name1)
with open(file_out1, 'wb') as outfile:
    pickle.dump(unitary_paritioning_of_Con_H, outfile)

file_name2 = 'experimental_ordering_updated_hamiltonians__{}.pickle'.format(unique_file_time)
file_out2=os.path.join(output_dir, file_name2)
with open(file_out2, 'wb') as outfile:
    pickle.dump(updated_hamiltonians, outfile)


print('pickle files dumped unqiue time id: {}'.format(unique_file_time))

########## sanity_check #########
# LCU check
E_LCU_dict ={}
for mol_name in unitary_paritioning_of_Con_H:
    
    N_Qubits= updated_hamiltonians[mol_name]['n_qubits']

    
    true_gs_energy = csvqe_results[mol_name][0]
    
    E_list=[]
    error_list=[]
    for n_qubits, tuple_fullH_ACset in enumerate(unitary_paritioning_of_Con_H[mol_name]):

        AC_set = tuple_fullH_ACset[1]
        if n_qubits == 0:
            Energy_I = list(AC_set[0][0].terms.values())[0] # <- no qubits!
            E_list.append(Energy_I) 
            error_list.append(abs(true_gs_energy-Energy_I))
        else:

            N_dict = {key:0 for key in AC_set}
            E_LCU = LCU_linalg_Energy(AC_set,
                                      N_dict,
                                      n_qubits,
                                      atol=1e-8,
                                      rtol=1e-05,
                                      check_reduction=False) ### <--- change for paper!
            E_list.append(copy(E_LCU))
            error_list.append(abs(true_gs_energy-E_LCU))
            del E_LCU
    E_LCU_dict[mol_name] = {'Energy_list': E_list,
                            'Error_list': error_list}
    

### save LCU energies!
file_name_LCU = 'E_LCU_all_EXP__{}.pickle'.format(unique_file_time)
file_out3=os.path.join(output_dir, file_name_LCU)
with open(file_out3, 'wb') as outfile:
    pickle.dump(E_LCU_dict, outfile)

# ### SeqRot check
# E_SeqRot_dict ={}
# for mol_name in unitary_paritioning_of_Con_H:
    
#     N_Qubits= updated_hamiltonians[mol_name]['n_qubits']

    
#     if N_Qubits> 10:
#         print(f'skipping: {mol_name} as qubit number too high')
#         continue

#     true_gs_energy = csvqe_results[mol_name][0]
    
#     E_list=[]
#     error_list=[]
#     for n_qubits, tuple_fullH_ACset in enumerate(unitary_paritioning_of_Con_H[mol_name]):

#         AC_set = tuple_fullH_ACset[1]
#         if n_qubits == 0:
#             Energy_I = list(AC_set[0][0].terms.values())[0] # <- no qubits!
#             E_list.append(Energy_I) 
#             error_list.append(abs(true_gs_energy-Energy_I))
#         else:

#             S_dict = {key:0 for key in AC_set}

#             E_SeqRot = SeqRot_linalg_Energy(AC_set,
#                                            S_dict,
#                                             n_qubits,
#                                              atol=1e-8,
#                                               rtol=1e-05,
#                                                check_reduction=False) ### <--- change for paper!


#             E_list.append(copy(E_SeqRot))
#             error_list.append(abs(true_gs_energy-E_SeqRot))
#             del E_SeqRot
#     E_SeqRot_dict[mol_name] = {'Energy_list': E_list,
#                             'Error_list': error_list}
    

# ### save SeqRot energies!
# file_name_SeqRot = 'E_SeqRot_all_EXP__{}.pickle'.format(unique_file_time)
# file_out3=os.path.join(output_dir, file_name_SeqRot)
# with open(file_out3, 'wb') as outfile:
#     pickle.dump(E_SeqRot_dict, outfile)


print('end time: {}'.format(datetime.datetime.now().strftime('%Y%b%d-%H%M%S%f')))
