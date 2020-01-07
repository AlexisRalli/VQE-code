from functools import reduce
from quchem.Unitary_partitioning import *
from quchem.Graph import *
from quchem.Tree_Functions import *

############### auto ###
from quchem.Hamiltonian_Generator_Functions import Hamiltonian
Molecule = 'LiH'
n_electrons = 3#10
num_shots = 10000
####

### Get Hamiltonian
Hamilt = Hamiltonian(Molecule,
                     run_scf = 1, run_mp2 = 1, run_cisd = 0, run_ccsd = 0, run_fci = 1,
                 basis = 'sto-3g',
                 multiplicity = 1,
                 geometry = None)

Hamilt.Get_all_info(get_FCI_energy=False)

List_PauliWords = Hamilt.QubitHamiltonianCompleteTerms
HamiltonainCofactors = Hamilt.HamiltonainCofactors
######################

##### Get Graph (finds anti_commuting_sets)
List_of_nodes = Get_PauliWords_as_nodes(List_PauliWords)
attribute_dictionary = {'Cofactors': HamiltonainCofactors}

List_of_nodes, node_attributes_dict = Get_list_of_nodes_and_attributes(List_of_nodes,
                                                                       attribute_dictionary=attribute_dictionary)

G = nx.Graph()
G = Build_Graph_Nodes(List_of_nodes, G, node_attributes_dict=node_attributes_dict, plot_graph=False)
G = Build_Graph_Edges_COMMUTING_QWC_AntiCommuting(G, List_of_nodes,'C', plot_graph = False)

# comp_G = Get_Complemenary_Graph(G, node_attributes_dict=node_attributes_dict, plot_graph=True) # <- not currently used


single_G, multi_G = Get_subgraphs(G, node_attributes_dict=node_attributes_dict)
s_colour = Colour_list_of_Graph(single_G, attribute_dictionary=attribute_dictionary, plot_graph=False,
                                strategy='largest_first')
m_colour = Colour_list_of_Graph(multi_G, attribute_dictionary=attribute_dictionary, plot_graph=False,
                                strategy='largest_first')

anti_commuting_sets = Get_unique_graph_colours(s_colour + m_colour)

######################


#S_dict, QWC_terms_to_measure = Get_all_S_terms(anti_commuting_sets, 'QWC')


# error for ANTI COMMUTING EXAMPLE!!!
S_dict, QWC_terms_to_measure = Get_all_S_terms(anti_commuting_sets, 'AC')
print(reduce(lambda count, LIST: count + len(LIST), QWC_terms_to_measure, 0))
QWC_terms_to_measure[-6]

# error in Find_Longest_tree

# def Find_Longest_tree(FILLED_anti_commuting_sets, max_set_size, anti_comm):
#     """
#      From a dictionary of fully anti-commuting sets (each entry is of same length, None used as filler. Finds
#      longests possible either fully anti-commuting or fully commuting subsets between rows #TODO need to find QWC too!
#
#
#     Args:
#         FILLED_anti_commuting_sets (dict): Dictionary of anti_commuting sets of PauliWords
#         max_set_size (int): Length of longest anti_commuting set
#         anti_comm_QWC (str): flags to find either:
#                                            qubit wise commuting (QWC) terms  -> flag = 'QWC',
#                                                              commuting terms -> flag = 'C',
#                                                         anti-commuting terms -> flag = 'AC'
#
#     Returns:
#         tree (dict): Dictionary of selected PauliWord: i_key = (index, row) and j_k = list of (index, row) either
#                     commuting or anti-commuting PauliWords (list)
#         best_combo (dict): selected PauliWord: i_key = (index, row) and j_k = list of (index, row) of best possible
#                           sub set
#
#     """
#     tree = {}
#     key_list = list(FILLED_anti_commuting_sets.keys())
#
#     # for key in FILLED_anti_commuting_sets:
#     for INDEX in tqdm(range(len(FILLED_anti_commuting_sets)), ascii=True, desc='Getting best Branch'):
#
#         running_best = 0
#         best_reduction_possible = len(key_list[INDEX:])  # <-- aka how many fully anti-commuting sets
#
#         key = key_list[INDEX]
#         selected_set = FILLED_anti_commuting_sets[key]
#         full_branch_key = []
#
#         for i in range(len(selected_set)):
#             P_word = selected_set[i]  # this will be the 'top' of the tree
#             branch_instance = []
#             branch_instance_holder = {}
#
#             if P_word is None:
#                 continue
#             else:
#                 branch_instance.append(str(*P_word.keys()))
#                 jk_list = []
#
#                 max_k = len(key_list[INDEX + 1:])  # remaining rows!
#
#                 for j in range(max_set_size):
#                     for k in key_list[INDEX + 1:][:-2]:
#                         for jj in range(max_set_size):
#                             P_comp = FILLED_anti_commuting_sets[k + 1][jj]
#
#                             print(k, jj)

#####
print(blah)
###

#
#
# # # putting anti_commuting_set dict in correct format for function!!! #TODO need to change original function! this is temporary fix!
# di = {}
# for key in anti_commuting_sets:
#     temp_l = []
#     for DICT in anti_commuting_sets[key]:
#         for P_Word in DICT:
#             temp_l.append((P_Word, DICT[P_Word]['Cofactors']))
#         di[key] = temp_l
#
#
# ### Ansatz
# from quchem.Ansatz_Generator_Functions import *
# qubits = cirq.LineQubit.range(Hamilt.MolecularHamiltonian.n_qubits)
# full_anstaz_circuit = cirq.Circuit.from_ops(cirq.X(qubits[0]), cirq.X(qubits[1]))
# # HF_initial_state= HF_state_generator(n_electrons, Hamilt.MolecularHamiltonian.n_qubits)
# # HF_UCC = Full_state_prep_circuit(HF_initial_state, T1_and_T2_theta_list=[])
# # HF_UCC.complete_UCC_circuit()
# # full_anstaz_circuit =HF_UCC.UCC_full_circuit
#
#
# zz = UnitaryPartition(di, full_anstaz_circuit, S_dict=S_dict)
# zz.Get_Quantum_circuits_and_constants()
# circuits_and_constants = zz.circuits_and_constants
# print('NUMBER of Q circuits = ',len(circuits_and_constants))
# print('Only need to perform = ', len(QWC_terms_to_measure), 'of them!')
# print('can measure the rest due to QWC')
#
# # these terms should QWC commute(85, 0), (92, 5)
# P1=circuits_and_constants[85]['PauliWord']
# P2=circuits_and_constants[92]['PauliWord']
# print(Commutativity(P1, P2, 'QWC'))
#
#
# PP1 = list(anti_commuting_sets[85][0].keys())[0]
# PP2 = list(anti_commuting_sets[92][5].keys())[0]
# print(Commutativity(PP1, PP2, 'QWC'))
#

