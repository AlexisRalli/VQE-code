from quchem.Hamiltonian_Generator_Functions import Hamiltonian
from quchem.Graph import BuildGraph_string
from quchem.Unitary_partitioning import *
from quchem.Ansatz_Generator_Functions import *
import numpy as np
import random



### Variable Parameters
Molecule = 'H2'
geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.74))]
n_electrons = 2
num_shots = 10000
max_iter = 50
T1_and_T2_theta_list_GUESS = [random.uniform(0, 2*math.pi) for i in range(3)]
####

### Get Hamiltonian
Hamilt = Hamiltonian(Molecule,
                     run_scf = 1, run_mp2 = 1, run_cisd = 0, run_ccsd = 0, run_fci = 1,
                 basis = 'sto-3g',
                 multiplicity = 1,
                 geometry = geometry) # normally None!

Hamilt.Get_all_info(get_FCI_energy=False)


##############

HF_initial_state= HF_state_generator(n_electrons, Hamilt.MolecularHamiltonian.n_qubits)

###### Build Graph --> to get anti-commuting sets

Commuting_indices = Hamilt.Commuting_indices
PauliWords = Hamilt.QubitHamiltonianCompleteTerms
constants = Hamilt.HamiltonainCofactors

HamiltGraph = BuildGraph_string(PauliWords, Commuting_indices, constants)
HamiltGraph.Build_string_nodes()  # plot_graph=True)
HamiltGraph.Build_string_edges()  # plot_graph=True)
HamiltGraph.Get_complementary_graph_string()  # plot_graph=True)
HamiltGraph.colouring(plot_graph=False)

anti_commuting_sets = HamiltGraph.anticommuting_sets


############ Running Optimization

from quchem.Scipy_Optimizer import *
NM = Optimizer(num_shots, T1_and_T2_theta_list_GUESS,
                  HF_initial_state, anti_commuting_sets,
                 noisy=True, store_values = True, optimized_result=None)
NM.get_env(max_iter)
#NM.plot_convergence()
print(NM.optimized_result)

from quchem.Misc_functions import *
Save_result_as_csv('unitary_parition_method', {'Energy': NM.obj_fun_values},
                   {'initial_angles': T1_and_T2_theta_list_GUESS, 'Molecule': Molecule, 'num_shots': num_shots, 'geometry': geometry}, folder='Results')



################# OLD APPROACH
from quchem.standard_method import *
PauliWords_and_constants = Get_PauliWord_strings_and_constant(Hamilt.QubitHamiltonianCompleteTerms, Hamilt.HamiltonainCofactors)




NM_standard = OptimizerSTANDARD(num_shots, T1_and_T2_theta_list_GUESS,
                  HF_initial_state, PauliWords_and_constants,
                 noisy=True, store_values = True, optimized_result=None)
NM_standard.get_env(max_iter)
#NM.plot_convergence()
print(NM_standard.optimized_result)



Save_result_as_csv('Standard_method', {'Energy': NM_standard.obj_fun_values},
                   {'initial_angles': T1_and_T2_theta_list_GUESS, 'Molecule': Molecule, 'num_shots': num_shots, 'geometry': geometry}, folder='Results')



#
# # x = circuits_and_constants[7]['circuit']
# # text_file = open("quantum_circuit.txt", "w")
# # n = text_file.write(x.to_text_diagram(transpose=True))
# # text_file.close()
