from quchem.LCU_method import *
from quchem.Hamiltonian_Generator_Functions import *

### Parameters
Molecule = 'H2'#LiH'
geometry = None
num_shots = 10000


### Get Hamiltonian
Hamilt = Hamiltonian(Molecule,
                     run_scf=1, run_mp2=1, run_cisd=1, run_ccsd=1, run_fci=1,
                     basis='sto-3g',
                     multiplicity=1,
                     geometry=geometry)  # normally None!

Hamilt.Get_Molecular_Hamiltonian()
SQ_CC_ops, THETA_params = Hamilt.Get_ia_and_ijab_terms(Coupled_cluser_param=True)
#print('UCC operations: ', SQ_CC_ops)

HF_transformations = Hamiltonian_Transforms(Hamilt.MolecularHamiltonian, SQ_CC_ops, Hamilt.molecule.n_qubits)

QubitHam = HF_transformations.Get_Qubit_Hamiltonian_JW()
#print('Qubit Hamiltonian: ', QubitHam)
QubitHam_PauliStr = HF_transformations.Convert_QubitMolecularHamiltonian_To_Pauliword_Str_list(QubitHam)

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

UCCSD_JW = UCCSD_Trotter_JW(SQ_CC_ops, THETA_params)
Second_Quant_CC_JW_OP_list = UCCSD_JW.SingleTrotterStep()

PauliWord_list = Convert_QubitOperator_To_Pauliword_Str_list(Second_Quant_CC_JW_OP_list)
HF_UCCSD_ansatz = Ansatz_Circuit(PauliWord_list, Hamilt.molecule.n_electrons, Hamilt.molecule.n_qubits)

PauliWord_list = Convert_QubitOperator_To_Pauliword_Str_list(Second_Quant_CC_JW_OP_list)
HF_UCCSD_ansatz = Ansatz_Circuit(PauliWord_list, Hamilt.molecule.n_electrons, Hamilt.molecule.n_qubits)

ansatz_Q_cicuit = HF_UCCSD_ansatz.Get_Full_HF_UCCSD_QC(THETA_params)

# ### Join Circuits
# S_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 1, 9: 1, 10: 1}
# w=ALCU_dict(ansatz_Q_cicuit, anti_commuting_set_stripped, S_dict, 4, 1)
# ####
#
# # simulate
# tt = ALCU_Simulation_Quantum_Circuit_Dict(w, 100, 1)
# tt.Get_expectation_value_via_parity()
# tt.Calc_energy_via_parity()
# tt.Energy
#
#
#
#
#
# S_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 1, 9: 1, 10: 1}
# def Calc_E_UP(THETA_params):
#     ansatz_Q_cicuit = HF_UCCSD_ansatz.Get_Full_HF_UCCSD_QC(THETA_params)
#     w = ALCU_dict(ansatz_Q_cicuit, anti_commuting_set_stripped, S_dict, 4, 1)
#     tt = ALCU_Simulation_Quantum_Circuit_Dict_SHOTS(w, 1000, 1)
#     tt.Get_expectation_value_via_parity()
#     tt.Calc_energy_via_parity()
#     return tt.Energy.real
# from quchem.Scipy_Optimizer import *
# THETA_params = [1, 2, 3]
# GG = Optimizer(Calc_E_UP, THETA_params, 'Nelder-Mead', store_values=True, display_iter_steps=True,
#                tol=1e-5,
#                display_convergence_message=True)
# GG.get_env(50)
# GG.plot_convergence()
# plt.show()


### Join Circuits
S_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
w, l1_norm =ALCU_dict(ansatz_Q_cicuit, anti_commuting_set_stripped, S_dict, 4, 1)
####

# simulate
tt = ALCU_Simulation_Quantum_Circuit_DictRAW(w, 1000, 1, l1_norm)
tt.Get_expectation_value_via_parity()
tt.Calc_energy_via_parity()
tt.Energy





S_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
def Calc_E_UP(THETA_params):
    ansatz_Q_cicuit = HF_UCCSD_ansatz.Get_Full_HF_UCCSD_QC(THETA_params)
    w, l1_norm = ALCU_dict(ansatz_Q_cicuit, anti_commuting_set_stripped, S_dict, 4, 1)
    tt = ALCU_Simulation_Quantum_Circuit_DictRAW(w, 1000, 1, l1_norm)
    tt.Get_expectation_value_via_parity()
    tt.Calc_energy_via_parity()
    return tt.Energy.real
from quchem.Scipy_Optimizer import *
THETA_params = [1, 2, 3]
# THETA_params = [0.23333333, 3.13333333, 3.05]
GG = Optimizer(Calc_E_UP, THETA_params, 'Nelder-Mead', store_values=True, display_iter_steps=True,
               tol=1e-5,
               display_convergence_message=True)
GG.get_env(50)
GG.plot_convergence()
plt.show()
