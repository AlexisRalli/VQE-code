import numpy as np
from functools import reduce
from quchem.Graph import *
#from itertools import zip_longest
from quchem.Tree_Functions import *

#### Manual ####

# List_PauliWords = [[(0, 'I'), (1, 'I'),(2, 'I'),(3, 'I'),(4, 'I'),(5, 'I'),(6, 'I'),(7, 'I'),(8, 'I'),(9, 'I'),(10, 'I'),(11, 'I')],
#                    [(0, 'Z'),(1, 'I'),(2, 'I'),(3, 'I'),(4, 'I'),(5, 'I'),(6, 'I'),(7, 'I'),(8, 'I'),(9, 'I'),(10, 'I'),(11, 'I')],
#                    [(0, 'Y'),(1, 'Z'),(2, 'Y'),(3, 'I'),(4, 'I'),(5, 'I'),(6, 'I'),(7, 'I'),(8, 'I'),(9, 'I'),(10, 'I'),(11, 'I')],
#                    [(0, 'X'),(1, 'Z'),(2, 'X'),(3, 'I'),(4, 'I'),(5, 'I'),(6, 'I'),(7, 'I'),(8, 'I'),(9, 'I'),(10, 'I'),(11, 'I')],
#                    [(0, 'Y'),(1, 'Z'),(2, 'Z'),(3, 'Z'),(4, 'Y'),(5, 'I'),(6, 'I'),(7, 'I'),(8, 'I'),(9, 'I'),(10, 'I'),(11, 'I')],
#                    [(0, 'X'),(1, 'Z'),(2, 'Z'),(3, 'Z'),(4, 'X'),(5, 'I'),(6, 'I'),(7, 'I'),(8, 'I'),(9, 'I'),(10, 'I'),(11, 'I')],
#                    [(0, 'Y'),(1, 'Z'),(2, 'Z'),(3, 'Z'),(4, 'Z'),(5, 'Z'),(6, 'Z'),(7, 'Z'),(8, 'Z'),(9, 'Z'),(10, 'Y'),(11, 'I')],
#                    [(0, 'X'),(1, 'Z'),(2, 'Z'),(3, 'Z'),(4, 'Z'),(5, 'Z'),(6, 'Z'),(7, 'Z'),(8, 'Z'),(9, 'Z'),(10, 'X'),(11, 'I')],
#                    [(0, 'I'),(1, 'Z'),(2, 'I'),(3, 'I'),(4, 'I'),(5, 'I'),(6, 'I'),(7, 'I'),(8, 'I'),(9, 'I'),(10, 'I'),(11, 'I')],
#                    [(0, 'I'),(1, 'Y'),(2, 'Z'),(3, 'Y'),(4, 'I'),(5, 'I'),(6, 'I'),(7, 'I'),(8, 'I'),(9, 'I'),(10, 'I'),(11, 'I')],
#                    [(0, 'I'),(1, 'X'),(2, 'Z'),(3, 'X'),(4, 'I'),(5, 'I'),(6, 'I'),(7, 'I'),(8, 'I'),(9, 'I'),(10, 'I'),(11, 'I')],
#                    [(0, 'I'),(1, 'Y'),(2, 'Z'),(3, 'Z'),(4, 'Z'),(5, 'Y'),(6, 'I'),(7, 'I'),(8, 'I'),(9, 'I'),(10, 'I'),(11, 'I')],
#                    [(0, 'I'),(1, 'X'),(2, 'Z'),(3, 'Z'),(4, 'Z'),(5, 'X'),(6, 'I'),(7, 'I'),(8, 'I'),(9, 'I'),(10, 'I'),(11, 'I')],
#                    [(0, 'I'),(1, 'Y'),(2, 'Z'),(3, 'Z'),(4, 'Z'),(5, 'Z'),(6, 'Z'),(7, 'Z'),(8, 'Z'),(9, 'Z'),(10, 'Z'),(11, 'Y')],
#                    [(0, 'I'),(1, 'X'),(2, 'Z'),(3, 'Z'),(4, 'Z'),(5, 'Z'),(6, 'Z'),(7, 'Z'),(8, 'Z'),(9, 'Z'),(10, 'Z'),(11, 'X')],
#                    [(0, 'I'),(1, 'I'),(2, 'Z'),(3, 'I'),(4, 'I'),(5, 'I'),(6, 'I'),(7, 'I'),(8, 'I'),(9, 'I'),(10, 'I'),(11, 'I')],
#                    [(0, 'I'),(1, 'I'),(2, 'Y'),(3, 'Z'),(4, 'Y'),(5, 'I'),(6, 'I'),(7, 'I'),(8, 'I'),(9, 'I'),(10, 'I'),(11, 'I')],
#                    [(0, 'I'),(1, 'I'),(2, 'X'),(3, 'Z'),(4, 'X'),(5, 'I'),(6, 'I'),(7, 'I'),(8, 'I'),(9, 'I'),(10, 'I'),(11, 'I')],
#                    [(0, 'I'),(1, 'I'),(2, 'Y'),(3, 'Z'),(4, 'Z'),(5, 'Z'),(6, 'Z'),(7, 'Z'),(8, 'Z'),(9, 'Z'),(10, 'Y'),(11, 'I')],
#                    [(0, 'I'),(1, 'I'),(2, 'X'),(3, 'Z'),(4, 'Z'),(5, 'Z'),(6, 'Z'),(7, 'Z'),(8, 'Z'),(9, 'Z'),(10, 'X'),(11, 'I')]
#                    ]
# HamiltonainCofactors = [(-3.9344419569678446+0j),
#  (1.04962640047693+0j),
#  (-0.023844584591133436+0j),
#  (-0.023844584591133436+0j),
#  (-0.026332990895885356+0j),
#  (-0.026332990895885356+0j),
#  (-0.017297109487008907+0j),
#  (-0.017297109487008907+0j),
#  (1.0496264004769302+0j),
#  (-0.023844584591133443+0j),
#  (-0.023844584591133443+0j),
#  (-0.026332990895885387+0j),
#  (-0.026332990895885387+0j),
#  (-0.01729710948700891+0j),
#  (-0.01729710948700891+0j),
#  (-0.09129805365197576+0j),
#  (-0.007987782352070982+0j),
#  (-0.007987782352070982+0j),
#  (-0.005200666861919969+0j),
#  (-0.005200666861919969+0j)]


############### auto ###

from quchem.Hamiltonian_Generator_Functions import Hamiltonian
Molecule = 'LiH'#'H2O'
#geometry =# [('H', (0., 0., 0.)), ('H', (0., 0., 0.74))]
n_electrons = 3#10
num_shots = 10000
####

### Get Hamiltonian
Hamilt = Hamiltonian(Molecule,
                     run_scf = 1, run_mp2 = 1, run_cisd = 0, run_ccsd = 0, run_fci = 1,
                 basis = 'sto-3g',
                 multiplicity = 1,
                 geometry = None) # normally None!

Hamilt.Get_all_info(get_FCI_energy=False)

List_PauliWords = Hamilt.QubitHamiltonianCompleteTerms
HamiltonainCofactors = Hamilt.HamiltonainCofactors

######################

#### Get Graph (finds anti_commuting_sets)

List_of_nodes = Get_PauliWords_as_nodes(List_PauliWords)
attribute_dictionary = {'Cofactors': HamiltonainCofactors}

List_of_nodes, node_attributes_dict = Get_list_of_nodes_and_attributes(List_of_nodes,
                                                                       attribute_dictionary=attribute_dictionary)

G = nx.Graph()
G = Build_Graph_Nodes(List_of_nodes, G, node_attributes_dict=node_attributes_dict, plot_graph=False)
G = Build_Graph_Edges_COMMUTING_QWC_AntiCommuting(G, List_of_nodes,'C', plot_graph = True)

# comp_G = Get_Complemenary_Graph(G, node_attributes_dict=node_attributes_dict, plot_graph=True) # <- not currently used


single_G, multi_G = Get_subgraphs(G, node_attributes_dict=node_attributes_dict)
s_colour = Colour_list_of_Graph(single_G, attribute_dictionary=attribute_dictionary, plot_graph=False,
                                strategy='largest_first')
m_colour = Colour_list_of_Graph(multi_G, attribute_dictionary=attribute_dictionary, plot_graph=False,
                                strategy='largest_first')

anti_commuting_sets = Get_unique_graph_colours(s_colour + m_colour)
print(anti_commuting_sets)

####


max_set_size = Get_longest_anti_commuting_set(anti_commuting_sets)
FILLED_anti_commuting_sets = Make_anti_commuting_sets_same_length(anti_commuting_sets, max_set_size)

tree_commute, best_combo_commute = Find_Longest_tree(FILLED_anti_commuting_sets, max_set_size, anti_comm= False)
#tree_anti, best_combo_anti = Find_Longest_tree(FILLED_anti_commuting_sets, max_set_size, anti_comm= True)


def Remaining_anti_commuting_sets(best_combo, anti_commuting_sets_RELATED_to_combo):
    missing_k = [k for k in anti_commuting_sets_RELATED_to_combo.keys() if
                 k not in [key for index, key in best_combo['j_k']] + [best_combo['i_key'][1]]]
    new_anti_commuting_sets = {}
    for key in missing_k:
        # new_anti_commuting_sets[key]=anti_commuting_sets[key]
        new_anti_commuting_sets[key] = anti_commuting_sets_RELATED_to_combo[key]
    return new_anti_commuting_sets


# # want to maximise the anti_commutativity between the remaining terms to do further Unitary partitioning!
# missing_k = [k for k in anti_commuting_sets.keys() if k not in [key for index, key in best_combo_commute['j_k']] + [best_combo_commute['i_key'][1]]]
#
# # only look at missing keys!!!
# new_anti_commuting_sets={}
# i=0
# for key in missing_k:
#     # new_anti_commuting_sets[key]=anti_commuting_sets[key]
#     new_anti_commuting_sets[i] = anti_commuting_sets[key]
#     i+=1

new_anti_commuting_sets = Remaining_anti_commuting_sets(best_combo_commute, anti_commuting_sets)
max_set_size = Get_longest_anti_commuting_set(new_anti_commuting_sets)
NEW_FILLED_anti_commuting_sets = Make_anti_commuting_sets_same_length(new_anti_commuting_sets, max_set_size)

# HERE can either look for next best commutative tree OR look for best anti_commuting terms to do further UP too!
tree_anti, best_combo_anti = Find_Longest_tree(NEW_FILLED_anti_commuting_sets, max_set_size, anti_comm= False)

print(best_combo_anti)

#missing_k_new = [k for k in new_anti_commuting_sets.keys() if k not in [key for index, key in best_combo_anti['j_k']] + [best_combo_anti['i_key'][1]]]
NEW_new_anti_commuting_sets = Remaining_anti_commuting_sets(best_combo_anti, NEW_FILLED_anti_commuting_sets)

NEW_max_set_size = Get_longest_anti_commuting_set(NEW_new_anti_commuting_sets)
NEW_NEW_FILLED_anti_commuting_sets = Make_anti_commuting_sets_same_length(NEW_new_anti_commuting_sets, max_set_size)
tree_anti, best_combo_anti = Find_Longest_tree(NEW_NEW_FILLED_anti_commuting_sets, NEW_max_set_size, anti_comm= False)
print(best_combo_anti)













# old idea (not working)
# # anti_commuting_sets key ordering... smallest first!
# dict_ordering_list = sorted(anti_commuting_sets, key=lambda k: len(anti_commuting_sets[k]), reverse=False)
# node_list = list(G.nodes)

# H = G.copy()
# #remove all of first set
# H.remove_nodes_from([P_word for sets in anti_commuting_sets[0] for P_word in sets])
#
# #remove all of second set BAR first term
# H.remove_nodes_from([P_word for sets in anti_commuting_sets[1][1::] for P_word in sets])
#
#
#
# # may be able to use max clique function (finds max no. of fully connected subgraphs!)
# # iterate through tree, selecting one P_word from each row
# # find max cliques
# list(nx.clique.find_cliques(H))
# min(list(nx.clique.find_cliques(H)))


####################

# # anti_commuting_sets key ordering... smallest first!
# dict_ordering_list = sorted(anti_commuting_sets, key=lambda k: len(anti_commuting_sets[k]), reverse=False)
#
# # get term with largest number of P_words
# max_num_terms = len(anti_commuting_sets[dict_ordering_list[-1]])

# LONGEST ANTI_COMMUTING SET