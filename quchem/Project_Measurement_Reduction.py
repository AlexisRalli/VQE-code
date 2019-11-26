#from .tests.VQE_methods.Hamiltonian_Generator_Functions import
from tests.VQE_methods.Hamiltonian_Generator_Functions import Hamiltonian
from tests.VQE_methods.Graph import BuildGraph_string
from tests.VQE_methods.Unitary_partitioning import *
from tests.VQE_methods.Ansatz_Generator_Functions import *
from tests.VQE_methods.quantum_circuit_functions import *
import numpy as np



### Variable Parameters
Molecule = 'H2'
geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.74))]
n_electrons = 2
num_shots = 10000
####

### Get Hamiltonian
Hamilt = Hamiltonian(Molecule,
                     run_scf = 1, run_mp2 = 1, run_cisd = 0, run_ccsd = 0, run_fci = 1,
                 basis = 'sto-3g',
                 multiplicity = 1,
                 geometry = geometry) # normally None!

Hamilt.Get_all_info(get_FCI_energy=False)


### Ansatz
HF_initial_state= HF_state_generator(n_electrons, Hamilt.MolecularHamiltonian.n_qubits)
#HF_initial_state = [0, 0, 1, 1]
#HF_initial_state = [0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]


HF_UCC = Full_state_prep_circuit(HF_initial_state, T1_and_T2_theta_list=[0,  1,  2])#, T1_and_T2_theta_list=[0,np.pi,0.5*np.pi]) // [-7.27091650e-05,  1.02335817e+00,  2.13607612e+00]
HF_UCC.complete_UCC_circuit()
full_anstaz_circuit =HF_UCC.UCC_full_circuit
#print(full_anstaz_circuit)




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

### Get Unitary Partition
zz = UnitaryPartition(anti_commuting_sets, full_anstaz_circuit, S=0)
zz.Get_Quantum_circuits_and_constants()
circuits_and_constants = zz.circuits_and_constants




from tests.VQE_methods.Simulating_Quantum_Circuit import *
xx = Simulation_Quantum_Circuit_Dict(circuits_and_constants, num_shots)
print(xx.Calc_energy_via_parity())




# from tests.VQE_methods.Scipy_Optimizer import *
# max_iter = 50
# NM = Optimizer(num_shots, [0,1,2],
#                   HF_initial_state,
#                  noisy=True, store_values = True, optimized_result=None)
# NM.get_env(max_iter)
# #NM.plot_convergence()
# print(NM.optimized_result)



################# OLD APPROACH
from tests.VQE_methods.standard_method import *
PauliWords_and_constants = Get_PauliWord_strings_and_constant(Hamilt.QubitHamiltonianCompleteTerms, Hamilt.HamiltonainCofactors)

# P_words_and_consts=[]
# for key in anti_commuting_sets:
#     for term in anti_commuting_sets[key]:
#         P_words_and_consts.append(term)

standard_dict = Get_quantum_circuits_and_constants_NORMAL(full_anstaz_circuit, PauliWords_and_constants)
yy = Simulation_Quantum_Circuit_Dict(standard_dict, num_shots)
print(yy.Calc_energy_via_parity())



# max_iter = 70
# theta_guess_list = [2.374, 1.437 , 1.163]
# NM = OptimizerSTANDARD(num_shots, theta_guess_list,
#                   HF_initial_state, PauliWords_and_constants,
#                  noisy=True, store_values = True, optimized_result=None)
# NM.get_env(max_iter)
# #NM.plot_convergence()
# print(NM.optimized_result)
#
#
# from tests.VQE_methods.Misc_functions import *
# #Save_result_as_csv('test', {'Energy': NM.obj_fun_values, 'initial_angles': theta_guess_list, 'Molecule': Molecule, 'num_shots': num_shots, 'geometry': geometry}, folder='Results')
# Save_result_as_csv('test', {'Energy': NM.obj_fun_values}, folder='Results')
#
#
#
# # x = circuits_and_constants[7]['circuit']
# # text_file = open("quantum_circuit.txt", "w")
# # n = text_file.write(x.to_text_diagram(transpose=True))
# # text_file.close()