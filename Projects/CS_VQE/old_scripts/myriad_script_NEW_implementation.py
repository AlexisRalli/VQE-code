import numpy as np
import cs_vqe as c
from copy import deepcopy as copy
from tqdm import tqdm
import pickle
import datetime

import quchem.Misc_functions.conversion_scripts as conv_scr
import cs_vqe_with_LCU as c_LCU

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



##### OLD WAY #### 
csvqe_results = {}
for speciesname in hamiltonians.keys():

    n_qubits = hamiltonians[speciesname][1]
    ham = hamiltonians[speciesname][2]
    ham_noncon = hamiltonians[speciesname][3]
    true_gs = hamiltonians[speciesname][4]

    print(speciesname,n_qubits)
    
    csvqe_out = c.csvqe_approximations_heuristic(ham, ham_noncon, n_qubits, true_gs)
    csvqe_results[speciesname] = csvqe_out
    print('  best order:',csvqe_out[3])
    print('  resulting errors:',csvqe_out[2],'\n')



####### SAVE OUTPUT details
unique_file_time = datetime.datetime.now().strftime('%Y%b%d-%H%M%S%f')
output_dir = os.path.join(working_dir, 'Pickle_out')
########


####### SAVE OUTPUT
file_name1 = 'standard_CS_VQE_exp__{}.pickle'.format(unique_file_time)
file_out1=os.path.join(output_dir, file_name1)
with open(file_out1, 'wb') as outfile:
    pickle.dump(csvqe_results, outfile)



##### NEW IMPLEMENTATION #### 
N_index=0
csvqe_results_NEW = {}

for speciesname in hamiltonians.keys():

    n_qubits = hamiltonians[speciesname][1]
    ham = hamiltonians[speciesname][2]
    ham_noncon = hamiltonians[speciesname][3]
    true_gs = hamiltonians[speciesname][4]

    print(speciesname,n_qubits)
    
    csvqe_out = c_LCU.csvqe_approximations_heuristic_LCU(
                               ham,
                               ham_noncon,
                               n_qubits, 
                               true_gs, 
                               N_index, 
                               check_reduction=False) ### <--- change for paper!

    csvqe_results_NEW[speciesname] = csvqe_out
    print('  best order:',csvqe_out[3])
    print('  resulting errors:',csvqe_out[2],'\n')





####### SAVE OUTPUT
file_name2 = 'NEW_method_CS_VQE_exp__{}.pickle'.format(unique_file_time)
file_out2=os.path.join(output_dir, file_name2)
with open(file_out2, 'wb') as outfile:
    pickle.dump(csvqe_results_NEW, outfile)


print('pickle files dumped unqiue time id: {}'.format(unique_file_time))

print('end time: {}'.format(datetime.datetime.now().strftime('%Y%b%d-%H%M%S%f')))