from quchem.Hamiltonian_Generator_Functions import *

### Variable Parameters
Molecule = 'H2'#LiH'
geometry = None
num_shots = 10000
HF_occ_index = [0,1] #[0, 1,2] # for occupied_orbitals_index_list
#######

### Get Hamiltonian
Hamilt = Hamiltonian(Molecule,
                     run_scf=1, run_mp2=1, run_cisd=1, run_ccsd=1, run_fci=1,
                     basis='sto-3g',
                     multiplicity=1,
                     geometry=geometry)  # normally None!

Hamilt.Get_Molecular_Hamiltonian()
SQ_CC_ops, THETA_params = Hamilt.Get_ia_and_ijab_terms(Coupled_cluser_param=True)
print('UCC operations: ', SQ_CC_ops)

HF_transformations = Hamiltonian_Transforms(Hamilt.MolecularHamiltonian, SQ_CC_ops, Hamilt.molecule.n_qubits)

QubitHam = HF_transformations.Get_Qubit_Hamiltonian_JW()
#print('Qubit Hamiltonian: ', QubitHam)
QubitHam_PauliStr = HF_transformations.Convert_QubitMolecularHamiltonian_To_Pauliword_Str_list(QubitHam)
print('Qubit Hamiltonian: ', QubitHam_PauliStr)

## calc energy via Lin. Alg.
# UCC_JW_excitation_matrix_list = HF_transformations.Get_Jordan_Wigner_CC_Matrices()
# HF_ref_ket, HF_ref_bra = Hamilt.Get_Basis_state_in_occ_num_basis(occupied_orbitals_index_list=HF_occ_index)
# w = CalcEnergy(Hamilt.MolecularHamiltonianMatrix, HF_ref_ket, Hamilt.molecule.n_qubits,
#                UCC_JW_excitation_matrix_list)
# w.Calc_HF_Energy()
# w.Calc_UCCSD_No_Trot(THETA_params)
# w.Calc_UCCSD_with_Trot(THETA_params)

### Ansatz ###
from quchem.Ansatz_Generator_Functions import *

UCCSD = UCCSD_Trotter_JW(SQ_CC_ops, THETA_params)
Second_Quant_CC_JW_OP_list = UCCSD.SingleTrotterStep()
PauliWord_list = Convert_QubitOperator_To_Pauliword_Str_list(Second_Quant_CC_JW_OP_list)
HF_UCCSD_ansatz = Ansatz_Circuit(PauliWord_list, Hamilt.molecule.n_electrons, Hamilt.molecule.n_qubits)
# THETA_params = [random.uniform(0, 2 * np.pi) for _ in range(Hamilt.num_theta_parameters)]
ansatz_Q_cicuit = HF_UCCSD_ansatz.Get_Full_HF_UCCSD_QC(THETA_params)
print(ansatz_Q_cicuit)



### Graph Colouring
from quchem.Graph import *

List_PauliWords, HamiltonainCofactors = zip(*QubitHam_PauliStr)

attribute_dictionary = {'Cofactors': HamiltonainCofactors}

List_of_nodes, node_attributes_dict = Get_list_of_nodes_and_attributes(List_PauliWords,
                                                                       attribute_dictionary=attribute_dictionary)

G = Hamiltonian_Graph(List_PauliWords, attribute_dictionary=attribute_dictionary)
anti_commuting_sets = G.Get_Pauli_grouping('AC', Graph_colouring_strategy='largest_first', plot_graph=False)
anti_commuting_sets = Get_PauliWord_constant_tuples(anti_commuting_sets, dict_str_label='Cofactors')

### Simulating Q Circuit

# Quantum Circuit dict
from quchem.Simulating_Quantum_Circuit import *
circuits_and_constants = Generate_Full_Q_Circuit_of_Molecular_Hamiltonian(ansatz_Q_cicuit, QubitHam_PauliStr, Hamilt.molecule.n_qubits)
xx = Simulation_Quantum_Circuit_Dict(circuits_and_constants, num_shots)
print(xx.Calc_energy_via_parity())

THETA_params = [0,1,2]
ansatz_Q_cicuit = HF_UCCSD_ansatz.Get_Full_HF_UCCSD_QC(THETA_params)
circuits_and_constants = Generate_Full_Q_Circuit_of_Molecular_Hamiltonian(ansatz_Q_cicuit, QubitHam_PauliStr, Hamilt.molecule.n_qubits)
xx = Simulation_Quantum_Circuit_Dict(circuits_and_constants, num_shots)
print(xx.Calc_energy_via_parity())

# ### Unitary Partitioning
from quchem.Unitary_partitioning import *
anti_commuting_set_stripped = Get_PauliWord_constant_tuples(anti_commuting_sets, dict_str_label='Cofactors')
zz = UnitaryPartition(anti_commuting_set_stripped, ansatz_Q_cicuit, S_dict=None) #{0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 1, 9: 1, 10: 1}
zz.Get_Quantum_circuits_and_constants()
circuits_and_constants = zz.circuits_and_constants
yy = Simulation_Quantum_Circuit_Dict(circuits_and_constants, num_shots)
print(yy.Calc_energy_via_parity())


### Simulating Q Circuit

def CalcE_QC(THETA_params):
    ansatz_Q_cicuit = HF_UCCSD_ansatz.Get_Full_HF_UCCSD_QC(THETA_params)
    circuits_and_constants = Generate_Full_Q_Circuit_of_Molecular_Hamiltonian(ansatz_Q_cicuit, QubitHam_PauliStr,
                                                                              Hamilt.molecule.n_qubits)
    xx = Simulation_Quantum_Circuit_Dict(circuits_and_constants, num_shots)
    E = xx.Calc_energy_via_parity()
    return E

from quchem.Scipy_Optimizer import *
THETA_params = [0,1,2]
GG = Optimizer(CalcE_QC, THETA_params, 'Nelder-Mead', store_values=True, display_iter_steps=True,  tol=1e-3,
               display_convergence_message= True)
GG.get_env(20)
GG.plot_convergence()


from quchem.Simulating_Quantum_Circuit import *
def Calculate_Gradient_by_QC(theta_guess_list):

    #HF_UCCSD_ansatz = Ansatz_Circuit(PauliWord_list, Hamilt.molecule.n_electrons, Hamilt.molecule.n_qubits)
    partial_gradient_list = []
    for j in range(len(theta_guess_list)):
        theta = theta_guess_list[j]


        theta_list_PLUS = theta_guess_list.copy()
        theta_list_PLUS[j] = theta_guess_list.copy()[j] + np.pi/4
        ansatz_Q_cicuit_PLUS = HF_UCCSD_ansatz.Get_Full_HF_UCCSD_QC(theta_list_PLUS)
        circuits_and_constants_PLUS = Generate_Full_Q_Circuit_of_Molecular_Hamiltonian(ansatz_Q_cicuit_PLUS, QubitHam_PauliStr,
                                                                                  Hamilt.molecule.n_qubits)
        E_Plus = Simulation_Quantum_Circuit_Dict(circuits_and_constants_PLUS, num_shots).Calc_energy_via_parity()

        theta_list_MINUS = theta_guess_list.copy()
        theta_list_MINUS[j] = theta_guess_list.copy()[j] - np.pi/4
        ansatz_Q_cicuit_MINUS = HF_UCCSD_ansatz.Get_Full_HF_UCCSD_QC(theta_list_MINUS)
        circuits_and_constants_MINUS = Generate_Full_Q_Circuit_of_Molecular_Hamiltonian(ansatz_Q_cicuit_MINUS, QubitHam_PauliStr,
                                                                                  Hamilt.molecule.n_qubits)
        E_MINUS = Simulation_Quantum_Circuit_Dict(circuits_and_constants_MINUS, num_shots).Calc_energy_via_parity()

        Gradient = (E_Plus - E_MINUS)  # /2
        partial_gradient_list.append((Gradient, theta))  # .append(Gradient)
    return partial_gradient_list

def Calc_Gradient_by_finite_differencing(theta_guess_list, delta=0.1):
    # gives very similar result when delta = 0.1 (#TODO think that when setting delta to very small value, errors in cirq simulation become important == wrong gradient given...
    partial_gradient_list = []
    for j in range(len(theta_guess_list)):
        theta = theta_guess_list[j]


        theta_list_PLUS = theta_guess_list.copy()
        theta_list_PLUS[j] = theta_guess_list.copy()[j] + delta/2
        ansatz_Q_cicuit_PLUS = HF_UCCSD_ansatz.Get_Full_HF_UCCSD_QC(theta_list_PLUS)
        circuits_and_constants_PLUS = Generate_Full_Q_Circuit_of_Molecular_Hamiltonian(ansatz_Q_cicuit_PLUS, QubitHam_PauliStr,
                                                                                  Hamilt.molecule.n_qubits)
        E_Plus = Simulation_Quantum_Circuit_Dict(circuits_and_constants_PLUS, num_shots).Calc_energy_via_parity()

        theta_list_MINUS = theta_guess_list.copy()
        theta_list_MINUS[j] = theta_guess_list.copy()[j] - delta/2
        ansatz_Q_cicuit_MINUS = HF_UCCSD_ansatz.Get_Full_HF_UCCSD_QC(theta_list_MINUS)
        circuits_and_constants_MINUS = Generate_Full_Q_Circuit_of_Molecular_Hamiltonian(ansatz_Q_cicuit_MINUS, QubitHam_PauliStr,
                                                                                  Hamilt.molecule.n_qubits)
        E_MINUS = Simulation_Quantum_Circuit_Dict(circuits_and_constants_MINUS, num_shots).Calc_energy_via_parity()
        Gradient = (E_Plus - E_MINUS)/delta
        partial_gradient_list.append((Gradient.real, theta))  # .append(Gradient)
    return partial_gradient_list


theta_guess_list = [1,2,3]
print(Calculate_Gradient_by_QC(theta_guess_list))
print(Calc_Gradient_by_finite_differencing(theta_guess_list, delta=0.1))

from quchem.TensorFlow_Opt import *
GG = Tensor_Flow_Optimizer(CalcE_QC, theta_guess_list, 'Adam', Calculate_Gradient_by_QC, learning_rate=0.1, beta1=0.9,
                           beta2=0.999, store_values=True, display_iter_steps=True)
GG.optimize(50)
GG.plot_convergence()


# THETA_params = [0,1,2]
# ansatz_Q_cicuit = HF_UCCSD_ansatz.Get_Full_HF_UCCSD_QC(THETA_params)
# circuits_and_constants = Generate_Full_Q_Circuit_of_Molecular_Hamiltonian(ansatz_Q_cicuit, QubitHam_PauliStr, Hamilt.molecule.n_qubits)
# xx = Simulation_Quantum_Circuit_Dict(circuits_and_constants, num_shots)
# print(xx.Calc_energy_via_parity())



### Get Unitary Partition
from quchem.Unitary_partitioning import *

def CalcE_QC_UP(THETA_params):
    ansatz_Q_cicuit = HF_UCCSD_ansatz.Get_Full_HF_UCCSD_QC(THETA_params)

    zz = UnitaryPartition(anti_commuting_sets, ansatz_Q_cicuit,
                          S_dict={0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 1, 9: 1,
                                  10: 1})
    zz.Get_Quantum_circuits_and_constants()
    circuits_and_constants = zz.circuits_and_constants

    xx = Simulation_Quantum_Circuit_Dict(circuits_and_constants, num_shots)
    E = xx.Calc_energy_via_parity()
    return E

THETA_params = [0,1,2]
GG = Optimizer(CalcE_QC_UP, THETA_params, 'Nelder-Mead', store_values=True, display_iter_steps=True,  tol=1e-3,
               display_convergence_message= True)
GG.get_env(50)
GG.plot_convergence()