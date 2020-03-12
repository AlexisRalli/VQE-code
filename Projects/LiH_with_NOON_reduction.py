from quchem.LCU_method import *
from quchem.Hamiltonian_Generator_Functions import *

### Parameters
Molecule = 'LiH'
geometry = None #[('Li', (0., 0., 0.)), ('H', (0., 0., 1.45))]
num_shots = 10000
basis = 'sto-3g'


### Get Hamiltonian
Hamilt = Hamiltonian(Molecule,
                     run_scf=1, run_mp2=1, run_cisd=1, run_ccsd=1, run_fci=1,
                     basis=basis,
                     multiplicity=1,
                     geometry=geometry)  # normally None!

Hamilt.Get_Molecular_Hamiltonian()
SQ_CC_ops, THETA_params = Hamilt.Get_ia_and_ijab_terms(Coupled_cluser_param=True)

SQ_CC_ops_REDUCED, THETA_params_REDUCED = Hamilt.Remove_NOON_terms(indices_to_remove_list_manual=[0,1,6,7])



HF_transformations = Hamiltonian_Transforms(Hamilt.MolecularHamiltonian, SQ_CC_ops_REDUCED, Hamilt.molecule.n_qubits)

QubitHam = HF_transformations.Get_Qubit_Hamiltonian_JW(threshold=None) # threshold=1e-12
#print('Qubit Hamiltonian: ', QubitHam)
QubitHam_PauliStr = HF_transformations.Convert_QubitMolecularHamiltonian_To_Pauliword_Str_list(QubitHam, Hamilt.molecule.n_qubits)

### Graph Colouring
from quchem.Graph import *
List_PauliWords, HamiltonainCofactors = zip(*QubitHam_PauliStr)
attribute_dictionary = {'Cofactors': HamiltonainCofactors}
List_of_nodes, node_attributes_dict = Get_list_of_nodes_and_attributes(List_PauliWords,
                                                                       attribute_dictionary=attribute_dictionary)
G = Hamiltonian_Graph(List_PauliWords, attribute_dictionary=attribute_dictionary)
anti_commuting_sets = G.Get_Pauli_grouping('AC', Graph_colouring_strategy='largest_first', plot_graph=False)

anti_commuting_set_stripped = Get_PauliWord_constant_tuples(anti_commuting_sets, dict_str_label='Cofactors')
print(anti_commuting_set_stripped)



#### LCU_GUG method

# test_set = anti_commuting_set_stripped[58]
test_set = anti_commuting_set_stripped[7]
S_index = 0
circuit = Complete_LCU_circuit(test_set, Hamilt.molecule.n_qubits, S_index)
# print(circuit)

### UCCSD ansatz
from quchem.Ansatz_Generator_Functions import *

UCCSD_JW = UCCSD_Trotter_JW(SQ_CC_ops_REDUCED, THETA_params_REDUCED)
Second_Quant_CC_JW_OP_list = UCCSD_JW.SingleTrotterStep()

PauliWord_list = Convert_QubitOperator_To_Pauliword_Str_list(Second_Quant_CC_JW_OP_list, Hamilt.molecule.n_qubits)
HF_UCCSD_ansatz = Ansatz_Circuit(PauliWord_list, Hamilt.molecule.n_electrons, Hamilt.molecule.n_qubits)

ansatz_Q_cicuit = HF_UCCSD_ansatz.Get_Full_HF_UCCSD_QC(THETA_params_REDUCED)

# if __name__ == '__main__':
#     THETA_params_REDUCED = [i for i in range(len(THETA_params_REDUCED))]
#
#     ansatz_Q_cicuit = HF_UCCSD_ansatz.Get_Full_HF_UCCSD_QC(THETA_params_REDUCED)
#     qubits_to_measure = (cirq.LineQubit(q_No) for q_No in range(Hamilt.molecule.n_qubits))
#     ansatz_Q_cicuit.append(cirq.measure(*qubits_to_measure))
#
#     # simulate
#     simulator = cirq.Simulator()
#     results = simulator.run(ansatz_Q_cicuit, repetitions=1000)
#     print(results.histogram(key='0,1,2,3,4,5,6,7,8,9,10,11'))  # Need key to match number of qubits!!!

### Join Circuits

# #TODO bug noticed... number of ancilla qubits changes per LCU dict... NEED TO FIX this!
# # currently loop to find max!
# a_list=[]
# for i in range(len(anti_commuting_set_stripped)):
#     if len(anti_commuting_set_stripped[i])>1:
#         print(i)
#         LCU_Dict = Get_R_linear_combination(anti_commuting_set_stripped[i], 0, Hamilt.molecule.n_qubits)
#         a_list.append(int(np.ceil(np.log2(len(LCU_Dict['R_LCU'])))))
# number_ancilla_qubits = max(a_list)

S_dict = {i:0 for i in range(len(anti_commuting_set_stripped))}
w=ALCU_dict(ansatz_Q_cicuit, anti_commuting_set_stripped, S_dict, Hamilt.molecule.n_qubits)
####

# simulate
tt = ALCU_Simulation_Quantum_Circuit_DictRAW(w, 1)
tt.Get_expectation_value_via_parity()
tt.Calc_energy_via_parity()
tt.Energy
