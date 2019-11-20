#from .tests.VQE_methods.Hamiltonian_Generator_Functions import
from tests.VQE_methods.Hamiltonian_Generator_Functions import Hamiltonian
from tests.VQE_methods.Graph import BuildGraph_string
from tests.VQE_methods.Unitary_partitioning import *
from tests.VQE_methods.Ansatz_Generator_Functions import *
from tests.VQE_methods.quantum_circuit_functions import *
import numpy as np


### Get Hamiltonian
Molecule = 'H2'
n_electrons = 2

Hamilt = Hamiltonian(Molecule,
                     run_scf = 1, run_mp2 = 1, run_cisd = 0, run_ccsd = 0, run_fci = 1,
                 basis = 'sto-3g',
                 multiplicity = 1,
                 geometry = None)

Hamilt.Get_all_info(get_FCI_energy=False)

# TODO write function to find HF state!



### Ansatz
HF_initial_state= HF_state_generator(n_electrons, Hamilt.MolecularHamiltonian.n_qubits)
#HF_initial_state = [0, 0, 1, 1]
#HF_initial_state = [0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]

# HF
HF_state_prep = State_Prep(HF_initial_state)
HF_state_prep_circuit = cirq.Circuit.from_ops(cirq.decompose_once(
    (HF_state_prep(*cirq.LineQubit.range(HF_state_prep.num_qubits())))))

# UCC

UCC = Full_state_prep_circuit(HF_initial_state, T1_and_T2_theta_list=[])#, T1_and_T2_theta_list=[0,np.pi,0.5*np.pi])
UCC.complete_UCC_circuit()
UCC_quantum_circuit =UCC.UCC_full_circuit
#print(UCC_quantum_circuit)


full_anstaz_circuit = cirq.Circuit.from_ops(
                                            [
                                            cirq.decompose_once(HF_state_prep_circuit),
                                            cirq.decompose_once(UCC_quantum_circuit)
                                            ]
                                            )

#print(full_anstaz_circuit)



Commuting_indices = Hamilt.Commuting_indices
PauliWords = Hamilt.QubitHamiltonianCompleteTerms
constants = Hamilt.HamiltonainCofactors

### Build Graph
HamiltGraph = BuildGraph_string(PauliWords, Commuting_indices, constants)
HamiltGraph.Build_string_nodes()  # plot_graph=True)
HamiltGraph.Build_string_edges()  # plot_graph=True)
HamiltGraph.Get_complementary_graph_string()  # plot_graph=True)
HamiltGraph.colouring(plot_graph=False)

anti_commuting_sets = HamiltGraph.anticommuting_sets

### Get Unitary Partition

All_X_sk_terms = X_sk_terms(anti_commuting_sets, S=0)
All_X_sk_terms.Get_all_X_sk_operator()

# print(All_X_sk_terms.normalised_anti_commuting_sets)
# print(All_X_sk_terms.X_sk_Ops)
#R_S_operators_by_key = Get_R_S_operators(All_X_sk_terms.X_sk_Ops)
# print(cirq.Circuit.from_ops(cirq.decompose_once(
#     (R_S_operators_by_key[7][0][0](*cirq.LineQubit.range(R_S_operators_by_key[7][0][0].num_qubits()))))))

circuits_and_constants = Get_quantum_circuits_and_constants(All_X_sk_terms, full_anstaz_circuit)
# circuits_and_constants={}
# for key in All_X_sk_terms.normalised_anti_commuting_sets:
#     if key not in All_X_sk_terms.X_sk_Ops:
#         PauliWord = All_X_sk_terms.normalised_anti_commuting_sets[key]['PauliWords'][0]
#         constant = All_X_sk_terms.normalised_anti_commuting_sets[key]['factor']
#
#         Pauli_circuit_object = Perform_PauliWord_and_Measure(PauliWord)
#         q_circuit_Pauliword = cirq.Circuit.from_ops(
#             cirq.decompose_once(
#                 (Pauli_circuit_object(*cirq.LineQubit.range(Pauli_circuit_object.num_qubits())))))
#         circuit_ops = list(q_circuit_Pauliword.all_operations())
#
#         if circuit_ops == []:
#             # deals with identity only circuit
#             circuits_and_constants[key] = {'circuit': None,
#                                            'factor': constant, 'PauliWord': PauliWord[0]}
#         else:
#             full_circuit = cirq.Circuit.from_ops(
#                 [
#                     *full_anstaz_circuit.all_operations(), # maybe make this a variable! (rather than repeated method)
#                     *circuit_ops
#                 ])
#
#             circuits_and_constants[key] = {'circuit': full_circuit,
#                                            'factor': constant, 'PauliWord': PauliWord[0]}
#
#     else:
#         term_reduction_circuits = [cirq.decompose_once(
#              (circuit(*cirq.LineQubit.range(circuit.num_qubits())))) for circuit, constant in R_S_operators_by_key[key]]
#
#         Pauliword_S = All_X_sk_terms.X_sk_Ops[key]['PauliWord_S']
#         q_circuit_Pauliword_S_object = Perform_PauliWord_and_Measure(Pauliword_S)
#
#         q_circuit_Pauliword_S = cirq.Circuit.from_ops(
#             cirq.decompose_once((q_circuit_Pauliword_S_object(*cirq.LineQubit.range(q_circuit_Pauliword_S_object.num_qubits())))))
#
#         full_circuit = cirq.Circuit.from_ops(
#             [
#                 *full_anstaz_circuit.all_operations(),      #maybe make this a variable! (rather than repeated method)
#                 *term_reduction_circuits,
#                 *q_circuit_Pauliword_S.all_operations()
#             ]
#         )
#
#         circuits_and_constants[key] = {'circuit': full_circuit, 'factor': Pauliword_S[1]*All_X_sk_terms.X_sk_Ops[key]['gamma_l'],
#                                        'PauliWord': Pauliword_S[0]}







# print(cirq.Circuit.from_ops(
#     [
#     cirq.decompose_once(T1_Ansatz_circuits[0][0](*cirq.LineQubit.range(T1_Ansatz_circuits[0][0].num_qubits()))),
#     cirq.decompose_once(
#             T1_Ansatz_circuits[0][1](*cirq.LineQubit.range(T1_Ansatz_circuits[0][1].num_qubits())))
#     ]
#             ))

from tests.VQE_methods.Simulating_Quantum_Circuit import *
num_shots = 1000
xx = Simulation_Quantum_Circuit_Dict(circuits_and_constants, num_shots)
print(xx.Calc_energy())


# from tests.VQE_methods.Scipy_Optimizer import *
# max_iter = 50
# NM = Optimizer(1000, [0,1,2],
#                   HF_state_prep_circuit, HF_initial_state, All_X_sk_terms,
#                  noisy=True, store_values = True, optimized_result=None)
# NM.get_env(max_iter)
# #NM.plot_convergence()
# print(NM.optimized_result)