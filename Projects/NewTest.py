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

UCCSD = UCCSD_Trotter(SQ_CC_ops, THETA_params)
Second_Quant_CC_JW_OP_list = UCCSD.SingleTrotterStep()
PauliWord_list = Convert_QubitOperator_To_Pauliword_Str_list(Second_Quant_CC_JW_OP_list)
HF_UCCSD_ansatz = Ansatz_Circuit(PauliWord_list, Hamilt.molecule.n_electrons, Hamilt.molecule.n_qubits)
# THETA_params = [random.uniform(0, 2 * np.pi) for _ in range(len(THETA_params))]
ansatz_Q_cicuit = HF_UCCSD_ansatz.Get_Full_HF_UCCSD_QC(THETA_params)
print(ansatz_Q_cicuit)



### Graph Colouring
from quchem.Graph import *

List_PauliWords, HamiltonainCofactors = zip(*QubitHam_PauliStr)

attribute_dictionary = {'Cofactors': HamiltonainCofactors}

List_of_nodes, node_attributes_dict = Get_list_of_nodes_and_attributes(List_PauliWords,
                                                                       attribute_dictionary=attribute_dictionary)

G = nx.Graph()
G = Build_Graph_Nodes(List_of_nodes, G, node_attributes_dict=node_attributes_dict, plot_graph=False)
G = Build_Graph_Edges_COMMUTING_QWC_AntiCommuting(G, List_of_nodes, 'C', plot_graph=False)

# comp_G = Get_Complemenary_Graph(G, node_attributes_dict=node_attributes_dict, plot_graph=True) # <- not currently used

single_G, multi_G = Get_subgraphs(G, node_attributes_dict=node_attributes_dict)
s_colour = Colour_list_of_Graph(single_G, attribute_dictionary=attribute_dictionary, plot_graph=False,
                                strategy='largest_first')
m_colour = Colour_list_of_Graph(multi_G, attribute_dictionary=attribute_dictionary, plot_graph=False,
                                strategy='largest_first')

anti_commuting_set = Get_unique_graph_colours(s_colour + m_colour)
print(anti_commuting_set)


### Simulating Q Circuit

# Quantum Circuit dict
circuits_and_constants = Generate_Full_Q_Circuit_of_Molecular_Hamiltonian(ansatz_Q_cicuit, QubitHam_PauliStr, Hamilt.molecule.n_qubits)
from quchem.Simulating_Quantum_Circuit import *
xx = Simulation_Quantum_Circuit_Dict(circuits_and_constants, num_shots)
print(xx.Calc_energy_via_parity())

THETA_params = [0,1,2]
ansatz_Q_cicuit = HF_UCCSD_ansatz.Get_Full_HF_UCCSD_QC(THETA_params)
circuits_and_constants = Generate_Full_Q_Circuit_of_Molecular_Hamiltonian(ansatz_Q_cicuit, QubitHam_PauliStr, Hamilt.molecule.n_qubits)
xx = Simulation_Quantum_Circuit_Dict(circuits_and_constants, num_shots)
print(xx.Calc_energy_via_parity())

# ### Unitary Partitioning
# from quchem.Unitary_partitioning import *
# zz = UnitaryPartition(anti_commuting_sets, ansatz_Q_cicuit, S_dict=None)
# zz.Get_Quantum_circuits_and_constants()
# circuits_and_constants = zz.circuits_and_constants
# yy = Simulation_Quantum_Circuit_Dict(circuits_and_constants, num_shots)
# print(yy.Calc_energy_via_parity())

from scipy.optimize import minimize

def CalcE_QC(THETA_params):
    ansatz_Q_cicuit = HF_UCCSD_ansatz.Get_Full_HF_UCCSD_QC(THETA_params)
    circuits_and_constants = Generate_Full_Q_Circuit_of_Molecular_Hamiltonian(ansatz_Q_cicuit, QubitHam_PauliStr,
                                                                              Hamilt.molecule.n_qubits)
    xx = Simulation_Quantum_Circuit_Dict(circuits_and_constants, num_shots)
    E = xx.Calc_energy_via_parity()
    print(E, THETA_params)
    return E


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

UCCSD = UCCSD_Trotter(SQ_CC_ops, THETA_params)
Second_Quant_CC_JW_OP_list = UCCSD.SingleTrotterStep()
PauliWord_list = Convert_QubitOperator_To_Pauliword_Str_list(Second_Quant_CC_JW_OP_list)
HF_UCCSD_ansatz = Ansatz_Circuit(PauliWord_list, Hamilt.molecule.n_electrons, Hamilt.molecule.n_qubits)
# THETA_params = [random.uniform(0, 2 * np.pi) for _ in range(len(THETA_params))]
ansatz_Q_cicuit = HF_UCCSD_ansatz.Get_Full_HF_UCCSD_QC(THETA_params)
print(ansatz_Q_cicuit)



### Graph Colouring
from quchem.Graph import *

List_PauliWords, HamiltonainCofactors = zip(*QubitHam_PauliStr)

attribute_dictionary = {'Cofactors': HamiltonainCofactors}

List_of_nodes, node_attributes_dict = Get_list_of_nodes_and_attributes(List_PauliWords,
                                                                       attribute_dictionary=attribute_dictionary)

G = nx.Graph()
G = Build_Graph_Nodes(List_of_nodes, G, node_attributes_dict=node_attributes_dict, plot_graph=False)
G = Build_Graph_Edges_COMMUTING_QWC_AntiCommuting(G, List_of_nodes, 'C', plot_graph=True)

# comp_G = Get_Complemenary_Graph(G, node_attributes_dict=node_attributes_dict, plot_graph=True) # <- not currently used

single_G, multi_G = Get_subgraphs(G, node_attributes_dict=node_attributes_dict)
s_colour = Colour_list_of_Graph(single_G, attribute_dictionary=attribute_dictionary, plot_graph=False,
                                strategy='largest_first')
m_colour = Colour_list_of_Graph(multi_G, attribute_dictionary=attribute_dictionary, plot_graph=False,
                                strategy='largest_first')

anti_commuting_set = Get_unique_graph_colours(s_colour + m_colour)
print(anti_commuting_set)


### Simulating Q Circuit

# Quantum Circuit dict
circuits_and_constants = Generate_Full_Q_Circuit_of_Molecular_Hamiltonian(ansatz_Q_cicuit, QubitHam_PauliStr, Hamilt.molecule.n_qubits)
from quchem.Simulating_Quantum_Circuit import *
xx = Simulation_Quantum_Circuit_Dict(circuits_and_constants, num_shots)
print(xx.Calc_energy_via_parity())

THETA_params = [0,1,2]
ansatz_Q_cicuit = HF_UCCSD_ansatz.Get_Full_HF_UCCSD_QC(THETA_params)
circuits_and_constants = Generate_Full_Q_Circuit_of_Molecular_Hamiltonian(ansatz_Q_cicuit, QubitHam_PauliStr, Hamilt.molecule.n_qubits)
xx = Simulation_Quantum_Circuit_Dict(circuits_and_constants, num_shots)
print(xx.Calc_energy_via_parity())

# ### Unitary Partitioning
# from quchem.Unitary_partitioning import *
# zz = UnitaryPartition(anti_commuting_sets, ansatz_Q_cicuit, S_dict=None)
# zz.Get_Quantum_circuits_and_constants()
# circuits_and_constants = zz.circuits_and_constants
# yy = Simulation_Quantum_Circuit_Dict(circuits_and_constants, num_shots)
# print(yy.Calc_energy_via_parity())

from scipy.optimize import minimize
def CalcE_QC(THETA_params):
    ansatz_Q_cicuit = HF_UCCSD_ansatz.Get_Full_HF_UCCSD_QC(THETA_params)
    circuits_and_constants = Generate_Full_Q_Circuit_of_Molecular_Hamiltonian(ansatz_Q_cicuit, QubitHam_PauliStr,
                                                                              Hamilt.molecule.n_qubits)
    xx = Simulation_Quantum_Circuit_Dict(circuits_and_constants, num_shots)
    E = xx.Calc_energy_via_parity()
    print(E, THETA_params)
    return E


THETA_params = [0,1,2]
res = minimize(CalcE_QC, THETA_params, method='nelder-mead',
               options={'xatol': 1e-8, 'disp': True})