import numpy as np
import cs_vqe as c
import ast
import os
from tqdm import tqdm

import cs_vqe_with_LCU as c_LCU

import pickle
import datetime


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



### run calc ##

N_index=0
check_reduction=False

csvqe_LCU_output={}
csvqe_standard_output={}

for mol_key in tqdm(hamiltonians.keys(), ascii=True, desc='running CS-VQE calcuations'):

    N_Qubits= hamiltonians[mol_key][1]

    ham = hamiltonians[mol_key][2] # full Hamiltonian
    true_gs = hamiltonians[mol_key][4] # ground state energy of full Hamiltonian (in Hartree)
    ham_noncon = hamiltonians[mol_key][3]  # noncontextual part of Hamiltonian, found by greedy DFS
    
    new_way = c_LCU.csvqe_approximations_heuristic_LCU(ham,
                               ham_noncon,
                               N_Qubits, 
                               true_gs, 
                               N_index, 
                               check_reduction=check_reduction)
    
    old_way = c.csvqe_approximations_heuristic(ham,
                               ham_noncon,
                               N_Qubits, 
                               true_gs)
    
    csvqe_LCU_output[mol_key] = new_way
    csvqe_standard_output[mol_key] = old_way



####### SAVE OUTPUT details
unique_file_time = datetime.datetime.now().strftime('%Y%b%d-%H%M%S%f')
output_dir = os.path.join(working_dir, 'Pickle_out')
########


####### SAVE OUTPUT
file_name1 = 'standard_CS_VQE_exp__{}.pickle'.format(unique_file_time)
file_out1=os.path.join(output_dir, file_name1)
with open(file_out1, 'wb') as outfile:
    pickle.dump(csvqe_standard_output, outfile)


file_name2 = 'LCU_CS_VQE_exp__{}.pickle'.format(unique_file_time)
file_out2=os.path.join(output_dir, file_name2)
with open(file_out2, 'wb') as outfile:
    pickle.dump(csvqe_standard_output, outfile)


print('pickle files dumped unqiue time id: {}'.format(unique_file_time))

print('end time: {}'.format(datetime.datetime.now().strftime('%Y%b%d-%H%M%S%f')))