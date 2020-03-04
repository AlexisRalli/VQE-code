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
LCU_Dict = Get_R_linear_combination(anti_commuting_set_stripped[9], 0, Hamilt.molecule.n_qubits)
number_ancilla_qubits = int(np.ceil(np.log2(len(LCU_Dict['R_LCU']))))

# S_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
# w, l1_norm =ALCU_dict(ansatz_Q_cicuit, anti_commuting_set_stripped, S_dict, Hamilt.molecule.n_qubits,
#                       number_ancilla_qubits)
# ####
#
# # simulate
# tt = ALCU_Simulation_Quantum_Circuit_DictRAW(w, 100, 1, l1_norm)
# tt.Get_expectation_value_via_parity()
# tt.Calc_energy_via_parity()
# tt.Energy
#
#
#
#
#
# S_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
# def Calc_E_UP(THETA_params):
#     ansatz_Q_cicuit = HF_UCCSD_ansatz.Get_Full_HF_UCCSD_QC(THETA_params)
#     w, l1_norm = ALCU_dict(ansatz_Q_cicuit, anti_commuting_set_stripped, S_dict, 4, 1)
#     tt = ALCU_Simulation_Quantum_Circuit_DictRAW(w, 500, 1, l1_norm)
#     tt.Get_expectation_value_via_parity()
#     tt.Calc_energy_via_parity()
#     return tt.Energy.real
# from quchem.Scipy_Optimizer import *
#
# import random
# THETA_params = [random.uniform(0,2*np.pi) for _ in range(len(THETA_params))]
#
#
# # THETA_params = [1, 2, 3]
# # THETA_params = [0.23333333, 3.13333333, 3.05]
# GG = Optimizer(Calc_E_UP, THETA_params, 'Nelder-Mead', store_values=True, display_iter_steps=True,
#                tol=1e-5,
#                display_convergence_message=True)
# GG.get_env(50)
# GG.plot_convergence()
# plt.show()
#


from quchem.quantum_circuit_functions import *
from quchem.Ansatz_Generator_Functions import HF_state_generator
HF_state_obj = HF_state_generator(Hamilt.molecule.n_electrons, Hamilt.molecule.n_qubits)
HF_state = HF_state_obj.Get_JW_HF_state_in_occ_basis()

def NEW_Anastz(theta):

    initial_state = State_Prep(HF_state)
    HF_circuit = cirq.Circuit(
        cirq.decompose_once((initial_state(*cirq.LineQubit.range(initial_state.num_qubits())))))

    Pauli = ('Y0 X1 X2 X3',  -1j)
    circuit_obj = full_exponentiated_PauliWord_circuit(Pauli, theta)
    circuit_exp = cirq.Circuit(
        cirq.decompose_once((circuit_obj(*cirq.LineQubit.range(circuit_obj.num_qubits())))))

    full_Circuit = cirq.Circuit(
       [
           HF_circuit.all_operations(),
           *circuit_exp.all_operations(),
       ]
    )

    return full_Circuit


ansatz_Q_cicuit = NEW_Anastz(np.pi)

S_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
w =ALCU_dict(ansatz_Q_cicuit, anti_commuting_set_stripped, S_dict, Hamilt.molecule.n_qubits,
                      number_ancilla_qubits)
###
# simulate
tt = ALCU_Simulation_Quantum_Circuit_DictRAW(w, 100, 1)
tt.Get_expectation_value_via_parity()
tt.Calc_energy_via_parity()
tt.Energy

S_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}



def Calc_E_UP(THETA):
    ansatz_Q_cicuit = NEW_Anastz(THETA)
    w = ALCU_dict(ansatz_Q_cicuit, anti_commuting_set_stripped, S_dict, 4, 1)
    tt = ALCU_Simulation_Quantum_Circuit_DictRAW(w, 500, 1)
    tt.Get_expectation_value_via_parity()
    tt.Calc_energy_via_parity()
    return tt.Energy.real
from quchem.Scipy_Optimizer import *

import random
THETA = random.uniform(0,2*np.pi)


# THETA_params = [1, 2, 3]
# THETA_params = [0.23333333, 3.13333333, 3.05]
GG = Optimizer(Calc_E_UP, [THETA], 'Nelder-Mead', store_values=True, display_iter_steps=True,
               tol=1e-5,
               display_convergence_message=True)
GG.get_env(50)
GG.plot_convergence()
plt.show()


E_list=[]
from tqdm import tqdm
for i in tqdm(np.arange(0, 2*np.pi, 0.1), ascii=True, desc='Getting ENERGIES'):

    ansatz_Q_cicuit = NEW_Anastz(i)
    w = ALCU_dict(ansatz_Q_cicuit, anti_commuting_set_stripped, S_dict, Hamilt.molecule.n_qubits,
                  number_ancilla_qubits)
    ###
    # simulate
    tt = ALCU_Simulation_Quantum_Circuit_DictRAW(w, 500, 1)
    tt.Get_expectation_value_via_parity()
    tt.Calc_energy_via_parity()
    E_list.append(tt.Energy)
plt.plot(E_list)
plt.show()