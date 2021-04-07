import numpy as np
import cs_vqe as c
from copy import deepcopy as copy
from quchem.Graph import *
from tqdm import tqdm
import pickle
import datetime
import conversion_scripts as conv_scr

from quchem.Unitary_partitioning_Seq_Rot import *
from quchem.Unitary_partitioning_LCU_method import * 
from scipy.sparse.linalg import expm
from openfermion import qubit_operator_sparse

import ast
import matplotlib.pyplot as plt
import os


# working_dir = os.getcwd()
working_dir = os.path.dirname(os.path.abspath(__file__)) # gets directory where running python file is!
hamiltonian_data = os.path.join(working_dir, 'hamiltonians.txt')
# hamiltonian_data = os.path.join(working_dir, 'Scratch/ContextualSubspaceVQE/hamiltonians.txt')
print('start time: {}'.format(datetime.datetime.now().strftime('%Y%b%d-%H%M%S%f')))
print('working directory:', working_dir)

with open(hamiltonian_data, 'r') as input_file:
    hamiltonians = ast.literal_eval(input_file.read())


for key in hamiltonians.keys():
    print(f"{key: <25}     n_qubits:  {hamiltonians[key][1]:<5.0f}")


## import data
csvqe_results_data = os.path.join(working_dir, 'csvqe_results.txt')
# hamiltonian_data = os.path.join(working_dir, 'Scratch/ContextualSubspaceVQE/csvqe_results.txt')
with open(csvqe_results_data, 'r') as input_file:
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
        Ham_Openf = conv_scr.Get_Operfermion_Hamiltonian(Ham)
        Contextual_Hamiltonian_list.append(list(Ham_Openf))
        qNo_list.append(qNo)
    
    updated_hamiltonians[key] = {
                                'encoding':hamiltonians[key][0],
                                'n_qubits': hamiltonians[key][1],
                                'full_H': conv_scr.Get_Operfermion_Hamiltonian(hamiltonians[key][2]),
#                                 'nonC_H_greedyDFS':Get_Operfermion_Hamiltonian(hamiltonians[key][3]),
                                'FCI':hamiltonians[key][4],                         
                                'gstate_noncon':hamiltonians[key][5], 
                                'Contextual_Hamiltonian_list': Contextual_Hamiltonian_list, #ham_out,   
                            'Contextual_Hamiltonian_qubitNo_list': qNo_list,  
                                'non_Contextual_H': conv_scr.Get_Operfermion_Hamiltonian(ham_noncon),# ham_noncon_out, 
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


for key in updated_hamiltonians.keys():
    
    H_con_List = updated_hamiltonians[key]['Contextual_Hamiltonian_list']
    
    
    anti_commuting_different_red_list=[]
    for H_con in H_con_List:
    
        Hamiltonian_graph_obj = Openfermion_Hamiltonian_Graph(H_con)
        anti_commuting_sets = Hamiltonian_graph_obj.Get_Clique_Cover_as_QubitOp(commutativity_flag, 
                                                                                Graph_colouring_strategy=Graph_colouring_strategy, 
                                                                                plot_graph=plot_graph)
        anti_commuting_different_red_list.append(anti_commuting_sets)
    
    H_con_reduced_and_antiC_set = list(zip(H_con_List, anti_commuting_different_red_list))
    
    unitary_paritioning_of_Con_H[key] = H_con_reduced_and_antiC_set



unique_file_time = datetime.datetime.now().strftime('%Y%b%d-%H%M%S%f')

file_name = 'experimental_ordering_anticommuting_vs_standard_conH__{}.pickle'.format(unique_file_time)
# raise ValueError(file_name)
with open(file_name, 'wb') as outfile:
    pickle.dump(unitary_paritioning_of_Con_H, outfile)

file_name2 = 'experimental_ordering_updated_hamiltonians__{}.pickle'.format(unique_file_time)
with open(file_name2, 'wb') as outfile:
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
with open(file_name_LCU, 'wb') as outfile:
    pickle.dump(E_LCU_dict, outfile)



### SeqRot check
E_SeqRot_dict ={}
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

            S_dict = {key:0 for key in AC_set}

            E_SeqRot = SeqRot_linalg_Energy(AC_set,
            								 S_dict,
            								  n_qubits,
            								   atol=1e-8,
            								    rtol=1e-05,
            								     check_reduction=False) ### <--- change for paper!


            E_list.append(copy(E_SeqRot))
            error_list.append(abs(true_gs_energy-E_SeqRot))
            del E_SeqRot
    E_SeqRot_dict[mol_name] = {'Energy_list': E_list,
                            'Error_list': error_list}
    

### save SeqRot energies!
file_name_SeqRot = 'E_SeqRot_all_EXP__{}.pickle'.format(unique_file_time)
with open(file_name_SeqRot, 'wb') as outfile:
    pickle.dump(E_SeqRot_dict, outfile)




print('end time: {}'.format(datetime.datetime.now().strftime('%Y%b%d-%H%M%S%f')))
