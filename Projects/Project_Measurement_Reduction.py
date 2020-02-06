from quchem.Hamiltonian_Generator_Functions import *

### Variable Parameters
Molecule = 'BeH2'#LiH'
geometry = None
num_shots = 10000
HF_occ_index = [0,1,2] #[0, 1,2] # for occupied_orbitals_index_list
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

QubitHam = HF_transformations.Get_Qubit_Hamiltonian_JW()
#print('Qubit Hamiltonian: ', QubitHam)
QubitHam_PauliStr = HF_transformations.Convert_QubitMolecularHamiltonian_To_Pauliword_Str_list(QubitHam)
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

G =  Hamiltonian_Graph(List_PauliWords, Graph_colouring_strategy='largest_first', attribute_dictionary=attribute_dictionary)
anti_commuting_sets = G.Get_Pauli_grouping('C', plot_graph=False)

# G = nx.Graph()
# G = Build_Graph_Nodes(List_of_nodes, G, node_attributes_dict=node_attributes_dict, plot_graph=False)
# G = Build_Graph_Edges_COMMUTING_QWC_AntiCommuting(G, List_of_nodes, 'C', plot_graph=False)
#
# # comp_G = Get_Complemenary_Graph(G, node_attributes_dict=node_attributes_dict, plot_graph=True) # <- not currently used
#
# single_G, multi_G = Get_subgraphs(G, node_attributes_dict=node_attributes_dict)
# s_colour = Colour_list_of_Graph(single_G, attribute_dictionary=attribute_dictionary, plot_graph=False,
#                                 strategy='largest_first')
# m_colour = Colour_list_of_Graph(multi_G, attribute_dictionary=attribute_dictionary, plot_graph=False,
#                                 strategy='largest_first')
#
# anti_commuting_sets = Get_unique_graph_colours(s_colour + m_colour)

anti_commuting_set_stripped = Get_PauliWord_constant_tuples(anti_commuting_sets, dict_str_label='Cofactors')
print(anti_commuting_set_stripped)

### NEXT graph!
set1_P, set1_C = zip(*anti_commuting_set_stripped[6])
set2_P, set2_C = zip(*anti_commuting_set_stripped[8])
# set1_P=[set1_P[0]]
# set1_C=[set1_C[0]]

NEW_attribute_dictionary = {'Cofactors': [*set1_C,*set2_C]}

List_of_nodes_NEW, node_attributes_dict_NEW = Get_list_of_nodes_and_attributes([*set1_P,*set2_P],
                                                                       attribute_dictionary=NEW_attribute_dictionary)

G_NEW =  Hamiltonian_Graph(List_of_nodes_NEW, Graph_colouring_strategy='largest_first', attribute_dictionary=NEW_attribute_dictionary)
commuting_sets = G_NEW.Get_Pauli_grouping('AC', plot_graph=False)

anti_comm_QWC_FLAG = 'C'
H = Graph_of_two_sets(set1_P, set2_P, anti_comm_QWC_FLAG, plot_graph=True)

adj_mat = nx.adjacency_matrix(H,nodelist=[*set1_P, *set2_P])
# I, J, Val = scipy.sparse.find(adj_mat[0:2,:])

Check_if_sets_completely_connected(H,set1_P, set2_P)


###CHECK in loop:
# checker=[]
# for i in range(len(anti_commuting_set_stripped)):
#     set1_P, set1_C = zip(*anti_commuting_set_stripped[i])
#     for j in np.arange(i+1,len(anti_commuting_set_stripped), 1):
#             set2_P, set2_C = zip(*anti_commuting_set_stripped[j])
#
#             if len(set2_P)>1: # checks if set2 is worth looking at!
#                 anti_comm_QWC_FLAG = 'C'
#                 H_NEW = Graph_of_two_sets(set1_P, set2_P, anti_comm_QWC_FLAG, plot_graph=False)
#                 checker.append(Check_if_sets_completely_connected(H_NEW,set1_P, set2_P))


###CHECK in loop:
full_connected_terms=np.zeros([len(anti_commuting_set_stripped), len(anti_commuting_set_stripped)])
num_success=[]
for i in range(len(anti_commuting_set_stripped)):

    if len(anti_commuting_set_stripped[i])>1:
        set1_P, set1_C = zip(*anti_commuting_set_stripped[i])

        for j in np.arange(i+1,len(anti_commuting_set_stripped), 1):
                set2_P, set2_C = zip(*anti_commuting_set_stripped[j])

                if len(set2_P)>1: # checks if set2 is worth looking at!
                    anti_comm_QWC_FLAG = 'C'
                    H_NEW = Graph_of_two_sets(set1_P, set2_P, anti_comm_QWC_FLAG, plot_graph=False)
                    if Check_if_sets_completely_connected(H_NEW,set1_P, set2_P):
                        full_connected_terms[i, j] = 1
                        num_success.append((i,j))

print('unique reduction:', len(set([i for i, j in num_success])))

index = 0
set1_P, set1_C = zip(*anti_commuting_set_stripped[num_success[index][0]])
set2_P, set2_C = zip(*anti_commuting_set_stripped[num_success[index][1]])
anti_comm_QWC_FLAG = 'C'
H = Graph_of_two_sets(set1_P, set2_P, anti_comm_QWC_FLAG, plot_graph=True)