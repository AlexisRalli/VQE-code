from quchem.Hamiltonian_Generator_Functions import *

### Variable Parameters
Molecule = 'LiH'#LiH'
geometry = None
num_shots = 10000
#######

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

QubitHam = HF_transformations.Get_Qubit_Hamiltonian_JW(threshold=None) # threshold=1e-12
#print('Qubit Hamiltonian: ', QubitHam)
QubitHam_PauliStr = HF_transformations.Convert_QubitMolecularHamiltonian_To_Pauliword_Str_list(QubitHam, Hamilt.molecule.n_qubits)
#print('Qubit Hamiltonian: ', QubitHam_PauliStr)

## calc energy via Lin. Alg.
# UCC_JW_excitation_matrix_list = HF_transformations.Get_Jordan_Wigner_CC_Matrices()
# HF_ref_ket, HF_ref_bra = Hamilt.Get_Basis_state_in_occ_num_basis(occupied_orbitals_index_list=HF_occ_index)
# w = CalcEnergy(Hamilt.MolecularHamiltonianMatrix, HF_ref_ket, Hamilt.molecule.n_qubits,
#                UCC_JW_excitation_matrix_list)
# w.Calc_HF_Energy()
# w.Calc_UCCSD_No_Trot(THETA_params)
# w.Calc_UCCSD_with_Trot(THETA_params)

### Ansatz ###
# from quchem.Ansatz_Generator_Functions import *
#
# UCCSD = UCCSD_Trotter(SQ_CC_ops, THETA_params)
# Second_Quant_CC_JW_OP_list = UCCSD.SingleTrotterStep()
# PauliWord_list = Convert_QubitOperator_To_Pauliword_Str_list(Second_Quant_CC_JW_OP_list)
# HF_UCCSD_ansatz = Ansatz_Circuit(PauliWord_list, Hamilt.molecule.n_electrons, Hamilt.molecule.n_qubits)
# # THETA_params = [random.uniform(0, 2 * np.pi) for _ in range(Hamilt.num_theta_parameters)]
# ansatz_Q_cicuit = HF_UCCSD_ansatz.Get_Full_HF_UCCSD_QC(THETA_params)
# print(ansatz_Q_cicuit)
#


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

# GET SUBGRAPH of graph:
relationship_between_Sets = 'C'
GRAPH_key_nodes = Get_subgraph_of_coloured_graph(anti_commuting_set_stripped, relationship_between_Sets)

print('No of edges: ', len(list(GRAPH_key_nodes.edges)))

# Get_clique cover of sub_graph
colour_key_for_nodes = Get_clique_cover(GRAPH_key_nodes, strategy='largest_first', plot_graph=False,
                                        node_attributes_dict=None)

print('reduction: ', len(anti_commuting_set_stripped) - len(colour_key_for_nodes))
print('unique: ', len(set([i[0] for i in list(GRAPH_key_nodes.edges)])))

### chekcing commutativity!:
# set1_P, set1_C = zip(*anti_commuting_set_stripped[105])
# set2_P, set2_C = zip(*anti_commuting_set_stripped[106])
#
# Graph_of_sets = Graph_of_two_sets(set1_P, set2_P,
#                                                   'C', plot_graph=False, node_attributes_dict=None)
# Check_if_sets_completely_connected(Graph_of_sets, set1_P, set2_P)