from quchem.LCU_method import *
from quchem.Hamiltonian_Generator_Functions import *

Molecule = 'LiH'  # LiH'
geometry = None
num_shots = 10000
HF_occ_index = [0, 1, 2]  # [0, 1,2] # for occupied_orbitals_index_list
#######

### Get Hamiltonian
Hamilt = Hamiltonian(Molecule,
                     run_scf=1, run_mp2=1, run_cisd=1, run_ccsd=1, run_fci=1,
                     basis='sto-3g',
                     multiplicity=1,
                     geometry=geometry)  # normally None!

Hamilt.Get_Molecular_Hamiltonian()
SQ_CC_ops, THETA_params = Hamilt.Get_ia_and_ijab_terms(Coupled_cluser_param=True)
# print('UCC operations: ', SQ_CC_ops)

HF_transformations = Hamiltonian_Transforms(Hamilt.MolecularHamiltonian, SQ_CC_ops, Hamilt.molecule.n_qubits)

QubitHam = HF_transformations.Get_Qubit_Hamiltonian_JW()
# print('Qubit Hamiltonian: ', QubitHam)
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