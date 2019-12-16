from quchem.Hamiltonian_Generator_Functions import Hamiltonian
from quchem.Graph import *
from quchem.Unitary_partitioning import *


### Variable Parameters
Molecule = 'LiH'#'LiH'
geometry = None
#n_electrons = 2
num_shots = 10000
####

### Get Hamiltonian
Hamilt = Hamiltonian(Molecule,
                     run_scf = 1, run_mp2 = 1, run_cisd = 0, run_ccsd = 0, run_fci = 1,
                 basis = 'sto-3g',
                 multiplicity = 1,
                 geometry = geometry) # normally None!


Hamilt.Get_Qubit_Hamiltonian_terms()

PauliWords = Hamilt.QubitHamiltonianCompleteTerms
constants = Hamilt.HamiltonainCofactors


### Get anti-commutativity graph
List_of_nodes = Get_PauliWords_as_nodes(PauliWords)
attribute_dictionary = {'Cofactors': constants}

List_of_nodes, node_attributes_dict = Get_list_of_nodes_and_attributes(List_of_nodes,
                                                                       attribute_dictionary=attribute_dictionary)

G = nx.Graph()
G = Build_Graph_Nodes(List_of_nodes, G, node_attributes_dict=node_attributes_dict, plot_graph=False)
G = Build_Graph_Edges_COMMUTING(G, List_of_nodes, plot_graph=False)

# comp_G = Get_Complemenary_Graph(G, node_attributes_dict=node_attributes_dict, plot_graph=True) # <- not currently used


single_G, multi_G = Get_subgraphs(G, node_attributes_dict=node_attributes_dict)
s_colour = Colour_list_of_Graph(single_G, attribute_dictionary=attribute_dictionary, plot_graph=False,
                                strategy='largest_first')
m_colour = Colour_list_of_Graph(multi_G, attribute_dictionary=attribute_dictionary, plot_graph=False,
                                strategy='largest_first')

anti_commuting_set = Get_unique_graph_colours(s_colour + m_colour)
print(anti_commuting_set)



graph_list=[]
graph_dict={}

for key in anti_commuting_set:
    other_keys = [other_key for other_key in range(len(anti_commuting_set)) if other_key != key]

    temp_dic_holder=[]

    for i in range(len(anti_commuting_set[key])):
        selected_PauliWord_node = anti_commuting_set[key][i][0]
        selected_PauliWord_constant = anti_commuting_set[key][i][1]['Cofactors']

        # for k in other_keys:
        #     comparison_nodes = [anti_commuting_set[k][j][0] for j in range(len(anti_commuting_set[k]))]
        #     comparison_constants = [anti_commuting_set[k][j][1]['Cofactors'] for j in range(len(anti_commuting_set[k]))]

        comparison_nodes = [anti_commuting_set[k][j][0] for k in other_keys for j in
                            range(len(anti_commuting_set[k]))]
        comparison_constants = [anti_commuting_set[k][j][1]['Cofactors'] for k in other_keys for j in
                                range(len(anti_commuting_set[k]))]

        List_of_nodes= [selected_PauliWord_node, *comparison_nodes]
        attribute_dictionary = {'Cofactors': [selected_PauliWord_constant, *comparison_constants]}

        List_of_nodes, node_attributes_dict = Get_list_of_nodes_and_attributes(List_of_nodes,
                                                                               attribute_dictionary=attribute_dictionary)

        H = nx.Graph()
        H = Build_Graph_Nodes(List_of_nodes, H, node_attributes_dict=node_attributes_dict, plot_graph=False)
        H = Build_Graph_Edges_COMMUTING(H, List_of_nodes, plot_graph=False)
        graph_list.append((selected_PauliWord_node, H))

        temp_dic_holder.append({'PauliWord': anti_commuting_set[key][i],
                         'graph': H})
    graph_dict[key]=temp_dic_holder


max_graphs={}
alt_graphs=[]
for key in graph_dict:
    max_connected = 0
    for i in range(len(graph_dict[key])):
        Graph = graph_dict[key][i]['graph']

       # max_connected = 0
        num_edges = len(Graph.edges)
        if  num_edges > max_connected:
            max_connected = num_edges
            max_graph = Graph
        elif num_edges == max_connected:
            alt_graphs.append((graph_dict[key][i]['PauliWord'],Graph))

    max_graphs[key] = (max_graph, graph_dict[key][i]['PauliWord'], alt_graphs)




# Graph= graph_dict[6][1]['graph']
# len(Graph.nodes)
#
#
# len(Graph.nodes)
# plt.figure()
# pos = nx.circular_layout(Graph)
# nx.draw(Graph, pos, with_labels=1)
# plt.show()




# single_terms_sets={}
# multi_term_sets={}
# for key in anti_commuting_sets:
#     if len(anti_commuting_sets[key]) == 1:
#         single_terms_sets[key] = anti_commuting_sets[key]
#     else:
#         multi_term_sets[key] = anti_commuting_sets[key]
#
#
#
# List_PauliWords = [[(0, 'I'), (1, 'I'), (2, 'I'), (3, 'I')],
#  [(0, 'Z'), (1, 'I'), (2, 'I'), (3, 'I')],
#  [(0, 'I'), (1, 'Z'), (2, 'I'), (3, 'I')],
#  [(0, 'I'), (1, 'I'), (2, 'Z'), (3, 'I')],
#  [(0, 'I'), (1, 'I'), (2, 'I'), (3, 'Z')],
#  [(0, 'Z'), (1, 'Z'), (2, 'I'), (3, 'I')],
#  [(0, 'Y'), (1, 'X'), (2, 'X'), (3, 'Y')],
#  [(0, 'Y'), (1, 'Y'), (2, 'X'), (3, 'X')],
#  [(0, 'X'), (1, 'X'), (2, 'Y'), (3, 'Y')],
#  [(0, 'X'), (1, 'Y'), (2, 'Y'), (3, 'X')],
#  [(0, 'Z'), (1, 'I'), (2, 'Z'), (3, 'I')],
#  [(0, 'Z'), (1, 'I'), (2, 'I'), (3, 'Z')],
#  [(0, 'I'), (1, 'Z'), (2, 'Z'), (3, 'I')],
#  [(0, 'I'), (1, 'Z'), (2, 'I'), (3, 'Z')],
#  [(0, 'I'), (1, 'I'), (2, 'Z'), (3, 'Z')]]
# Node_and_connected_Nodes = [(0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
#              (1, [0, 2, 3, 4, 5, 10, 11, 12, 13, 14]),
#              (2, [0, 1, 3, 4, 5, 10, 11, 12, 13, 14]),
#              (3, [0, 1, 2, 4, 5, 10, 11, 12, 13, 14]),
#              (4, [0, 1, 2, 3, 5, 10, 11, 12, 13, 14]),
#              (5, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
#              (6, [0, 5, 7, 8, 9, 10, 11, 12, 13, 14]),
#              (7, [0, 5, 6, 8, 9, 10, 11, 12, 13, 14]),
#              (8, [0, 5, 6, 7, 9, 10, 11, 12, 13, 14]),
#              (9, [0, 5, 6, 7, 8, 10, 11, 12, 13, 14]),
#              (10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]),
#              (11, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14]),
#              (12, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]),
#              (13, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14]),
#              (14, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])]
# HamiltonainCofactors = [(-0.32760818995565577 + 0j),
#                         (0.1371657293179602 + 0j),
#                         (0.1371657293179602 + 0j),
#                         (-0.13036292044009176 + 0j),
#                         (-0.13036292044009176 + 0j),
#                         (0.15660062486143395 + 0j),
#                         (0.04919764587885283 + 0j),
#                         (-0.04919764587885283 + 0j),
#                         (-0.04919764587885283 + 0j),
#                         (0.04919764587885283 + 0j),
#                         (0.10622904488350779 + 0j),
#                         (0.15542669076236065 + 0j),
#                         (0.15542669076236065 + 0j),
#                         (0.10622904488350779 + 0j),
#                         (0.1632676867167479 + 0j)]
#
# List_of_nodes = Get_PauliWords_as_nodes(List_PauliWords)
# attribute_dictionary = {'Cofactors': HamiltonainCofactors, 'random_attribute': [i for i in range(len(HamiltonainCofactors))]}
#
# List_of_nodes, node_attributes_dict = Get_list_of_nodes_and_attributes(List_of_nodes,
#                                                                        attribute_dictionary=attribute_dictionary)
#
# G = nx.Graph()
# G = Build_Graph_Nodes(List_of_nodes, G, node_attributes_dict=node_attributes_dict, plot_graph=False)
# G = Build_Graph_Edges(G, Node_and_connected_Nodes, plot_graph = True)
#
# #comp_G = Get_Complemenary_Graph(G, node_attributes_dict=node_attributes_dict, plot_graph=True) # <- not currently used
#
#
# single_G, multi_G = Get_subgraphs(G, node_attributes_dict=node_attributes_dict)
# s_colour = Colour_list_of_Graph(single_G, attribute_dictionary=attribute_dictionary, plot_graph=False,
#                                 strategy='largest_first')
# m_colour = Colour_list_of_Graph(multi_G, attribute_dictionary=attribute_dictionary, plot_graph=False,
#                                 strategy='largest_first')
#
# anti_commuting_set = Get_unique_graph_colours(s_colour + m_colour)
# print(anti_commuting_set)
