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

max_set_size = Get_longest_anti_commuting_set(anti_commuting_sets)
FILLED_anti_commuting_sets = Make_anti_commuting_sets_same_length(anti_commuting_sets, max_set_size)

tree_commute, best_combo_commute = Find_Longest_tree(FILLED_anti_commuting_sets, max_set_size, 'C')


new_anti_commuting_sets = Remaining_anti_commuting_sets(best_combo_commute, anti_commuting_sets)
max_set_size = Get_longest_anti_commuting_set(new_anti_commuting_sets)
NEW_FILLED_anti_commuting_sets = Make_anti_commuting_sets_same_length(new_anti_commuting_sets, max_set_size)

tree_anti, best_combo_anti = Find_Longest_tree(NEW_FILLED_anti_commuting_sets, max_set_size, 'AC')


# FIRST REDUCTION
S_dict = {}
S_dict[best_combo_commute['i_key'][1]] = best_combo_commute['i_key'][0]
for index, key in best_combo_commute['j_k']: #, :
    S_dict[key]= index

# SECOND REDUCTION
S_dict[best_combo_anti['i_key'][1]] = best_combo_anti['i_key'][0]
for index, key in best_combo_commute['j_k']: #, :
    S_dict[key]= index

# REMAINING TERMS
for key in anti_commuting_sets.keys():
    if key not in S_dict.keys():
        S_dict[key] = 0







NEW_anti_commuting_sets = {}
for key in S_dict:
    NEW_anti_commuting_sets[key] = anti_commuting_sets[key][S_dict[key]]

List_PauliWords=[]
HamiltonainCofactors=[]

for key in NEW_anti_commuting_sets:
    for P_Word, DICT in NEW_anti_commuting_sets[key].items():
        List_PauliWords.append(P_Word)
        HamiltonainCofactors.append(DICT['Cofactors'])


List_of_nodes = List_PauliWords
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


##### EXTRA reduction

# max_set_size = Get_longest_anti_commuting_set(anti_commuting_sets)
# FILLED_anti_commuting_sets = Make_anti_commuting_sets_same_length(anti_commuting_sets, max_set_size)
# tree_commute, best_combo_commute = Find_Longest_tree(FILLED_anti_commuting_sets, max_set_size, anti_comm= True)
#
#
# # FIRST REDUCTION
# S_dict = {}
# S_dict[best_combo_commute['i_key'][1]] = best_combo_commute['i_key'][0]
# for index, key in best_combo_commute['j_k']:
#     S_dict[key]= index
#
# # REMAINING TERMS
# for key in anti_commuting_sets.keys():
#     if key not in S_dict.keys():
#         S_dict[key] = 0


#
#
# putting anti_commuting_set dict in correct format for function!!! #TODO need to change original function! this is temporary fix!

di = {}
for key in anti_commuting_sets:
    temp_l = []
    for DICT in anti_commuting_sets[key]:
        for P_Word in DICT:
            temp_l.append((P_Word, DICT[P_Word]['Cofactors']))
        di[key] = temp_l



### Ansatz
from quchem.Ansatz_Generator_Functions import *
qubits = cirq.LineQubit.range(Hamilt.MolecularHamiltonian.n_qubits)
full_anstaz_circuit = cirq.Circuit.from_ops(cirq.X(qubits[0]), cirq.X(qubits[1]))
# HF_initial_state= HF_state_generator(n_electrons, Hamilt.MolecularHamiltonian.n_qubits)
# HF_UCC = Full_state_prep_circuit(HF_initial_state, T1_and_T2_theta_list=[])
# HF_UCC.complete_UCC_circuit()
# full_anstaz_circuit =HF_UCC.UCC_full_circuit


zz = UnitaryPartition(di, full_anstaz_circuit, S=S_dict)
zz.Get_Quantum_circuits_and_constants()
circuits_and_constants = zz.circuits_and_constants
print('NUMBER of Q circuits = ',len(circuits_and_constants))

# new_anti_commuting_sets = Remaining_anti_commuting_sets(best_combo_commute, anti_commuting_sets)
# max_set_size = Get_longest_anti_commuting_set(new_anti_commuting_sets)
# NEW_FILLED_anti_commuting_sets = Make_anti_commuting_sets_same_length(new_anti_commuting_sets, max_set_size)