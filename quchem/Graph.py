import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm



def Get_PauliWords_as_nodes(List_PauliWords):
    """

    Function takes in list of PauliWords and returns a list of PauliWords in string format

    Args:
        PauliWords (list): A list of PauliWords, which are tuples of (qubitNo(int), qubitOp(str))

    Returns:
        List_of_PauliWord_strings (list): list of PauliWords as strings


    .. code-block:: python

       from quchem.Graph import *


        PauliWords = [
                         [(0, 'I'), (1, 'I'), (2, 'I'), (3, 'I')],
                         [(0, 'Z'), (1, 'I'), (2, 'I'), (3, 'I')],
                         [(0, 'I'), (1, 'Z'), (2, 'I'), (3, 'I')],
                         [(0, 'I'), (1, 'I'), (2, 'I'), (3, 'Z')],
                         [(0, 'I'), (1, 'I'), (2, 'Z'), (3, 'Z')]
                     ]

        Get_PauliWords_as_nodes(List_PauliWords)
        >> ['I0 I1 I2 I3', 'Z0 I1 I2 I3', 'I0 Z1 I2 I3', 'I0 I1 I2 Z3', 'I0 I1 Z2 Z3']
    """

    List_of_PauliWord_strings = []

    for PauliWord in List_PauliWords:
        PauliStrings = ['{}{}'.format(qubitOp, qubitNo) for qubitNo, qubitOp in PauliWord]
        seperator = ' '
        List_of_PauliWord_strings.append(seperator.join(PauliStrings))
    return List_of_PauliWord_strings

def Get_list_of_nodes_and_attributes(List_of_nodes, attribute_dictionary):
    """

    Function builds nodes of graph with attributes

    IMPORTANT the index of a node in list of nodes... must match the index in the attributes list in attribute dictionary!

    Args:
        List_of_nodes (list): A list of strings or ints that you want to have as nodes
        attribute_dictionary (dict): A dictionary containing node parameters. IMPORTANT indexing must match
                                                node list.

    Returns:



    .. code-block:: python

       from quchem.Graph import *

        List_of_nodes =  [
                            'I0 I1 I2 I3',
                            'Z0 I1 I2 I3',
                            'I0 Z1 I2 I3',
                            'I0 I1 I2 Z3',
                            'I0 I1 Z2 Z3'
                        ]

       attribute_dictionary =  {
                                'Cofactors': [(-0.32760818995565577+0j),
                                              (0.1371657293179602+0j),
                                              (0.1371657293179602+0j),
                                              (-0.13036292044009176+0j),
                                              (0.1632676867167479+0j)],
                                'random_attribute': [0, 1, 2, 3, 4]
                                }

        Get_list_of_nodes_and_attributes(List_of_nodes, attribute_dictionary)
        >> {'I0 I1 I2 I3': {
                            'Cofactors': (-0.32760818995565577+0j),
                             'random_attribute': 0
                            },
         'Z0 I1 I2 I3': {
                        'Cofactors': (0.1371657293179602+0j),
                        'random_attribute': 1
                        },

         'I0 Z1 I2 I3': {
                         'Cofactors': (0.1371657293179602+0j),
                          'random_attribute': 2
                        },

         'I0 I1 I2 Z3': {
                         'Cofactors': (-0.13036292044009176+0j),
                          'random_attribute': 3
                       },

         'I0 I1 Z2 Z3': {
                         'Cofactors': (0.1632676867167479+0j),
                          'random_attribute': 4
                          }
          }

    """

    node_attributes_dict = {node: None for node in List_of_nodes}

    for i in range(len(List_of_nodes)):
        node = List_of_nodes[i]
        temp = {}
        for attribute_label in attribute_dictionary:
            temp.update({attribute_label: attribute_dictionary[attribute_label][i]})
        node_attributes_dict[node] = temp

    return List_of_nodes, node_attributes_dict

def Build_Graph_Nodes(List_of_nodes, Graph, node_attributes_dict=None, plot_graph=False):
    """

    Function builds nodes of graph with attributes

    Args:
        List_of_nodes (list): A list of Pauliwords, where each entry is a tuple of (PauliWord, constant)
        Graph ():
        node_attributes_dict
        plot_graph (optional, bool): whether to plot graph

    Returns:



    .. code-block:: python

       from quchem.Graph import *

       node_attributes_dict =  {'Cofactor': {'I0 I1 I2 I3': (-0.09706626861762624+0j),
                                                         'Z0 I1 I2 I3': (0.17141282639402405+0j),
                                                         'I0 Z1 I2 I3': (0.171412826394024+0j),
                                                         'I0 I1 Z2 I3': (-0.2234315367466397+0j),
                                                         'I0 I1 I2 Z3': (-0.2234315367466397+0j),
                                                         'Z0 Z1 I2 I3': (0.1686889816869329+0j),
                                                         'Y0 X1 X2 Y3': (0.04530261550868928+0j),
                                                         'Y0 Y1 X2 X3': (-0.04530261550868928+0j),
                                                         'X0 X1 Y2 Y3': (-0.04530261550868928+0j),
                                                         'X0 Y1 Y2 X3': (0.04530261550868928+0j),
                                                         'Z0 I1 Z2 I3': (0.12062523481381837+0j),
                                                         'Z0 I1 I2 Z3': (0.16592785032250768+0j),
                                                         'I0 Z1 Z2 I3': (0.16592785032250768+0j),
                                                         'I0 Z1 I2 Z3': (0.12062523481381837+0j),
                                                         'I0 I1 Z2 Z3': (0.174412876106516+0j)
                                                         }
                                                }

        DO SOMETHING
        >> blah

    """
    for node in List_of_nodes:
        Graph.add_node(node)

    if node_attributes_dict is not None:
        nx.set_node_attributes(Graph, node_attributes_dict)

    if plot_graph == True:
        plt.figure()
        nx.draw(Graph, with_labels=1)
        plt.show()
    return Graph

def Build_Graph_Edges_defined_by_indices(Graph, Node_and_connected_Nodes, plot_graph = False):
    """

    Function builds graph edges from defined node indices!

    e.g.
    [   (0, [1,3]),
        (1, [0,3]),
        (2, []),
        (3, [0, 1])
    ]

    first part of tuple is defined node and other part is list of nodes to connect too

    Args:
        Node_and_connected_Nodes (list): list of tuples, where each tuple is:
                                    (defined_node_index, [list of node indices that commute with defined_node_index])
        Graph: networkX graph with nodes already defined
        plot_graph (optional, bool): whether to plot graph

    Returns:
        Graph: Graph with nodes connected by defined indices

    """
    nodes_list = list(Graph.nodes())
    for node_index, connected_node_indices in Node_and_connected_Nodes:
        for index in connected_node_indices:
            if index != []:
                Graph.add_edge(nodes_list[node_index], nodes_list[index])

    if plot_graph == True:
        plt.figure()
        pos = nx.circular_layout(Graph)
        nx.draw(Graph, pos, with_labels=1)
        plt.show()
    return Graph

def Commutativity(P1, P2, anti_comm_QWC):
    """

     Find if two PauliWords either commute or anti_commute.
     By default it will check if they commute.

    Args:
        P1 (str): First PauliWord to compare
        P2 (str): Second PauliWord to compare
        anti_comm_QWC (str): flags to find either:
                                                   qubit wise commuting (QWC) terms  -> flag = 'QWC',
                                                                     commuting terms -> flag = 'C',
                                                                anti-commuting terms -> flag = 'AC'

    Returns:
        (bool): True or false as to whether terms commute or anti_commute

    .. code-block:: python
       :emphasize-lines: 6

       from quchem.Tree_Fucntions import *

        P1 = 'X0 X1 X2 X3 X4'
        P2 = 'I0 I1 I2 Z3 Y4'

        Commutativity(P1, P2, anti_comm_QWC='C')
       >> True

       Commutativity(P1, P2, anti_comm_QWC='QWC')
       >> False

       Commutativity(P1, P2, anti_comm_QWC='AC')
       >> False

    """
    P1 = P1.split(' ')
    P2 = P2.split(' ')

    checker = np.zeros(len(P1))
    for i in range(len(P1)):
        if P1[i][0] == P2[i][0]:
            checker[i] = 1
        elif P1[i][0] == 'I' or P2[i][0] == 'I':
            checker[i] = 1
        else:
            checker[i] = -1

    if anti_comm_QWC == 'QWC':
        # QWC commuting
        if bool(np.all([x==1 for x in checker])) is True:
            return True
        else:
            return False
    else:

        if anti_comm_QWC == 'C':
            # Commuting
            if np.prod(checker) == 1:
                return True
            else:
                return False
        elif anti_comm_QWC == 'AC':
            # ANTI-commuting
            if np.prod(checker) == -1:
                return True
            else:
                return False
        else:
            raise KeyError('Incorrect flag used. anti_comm_QWC must be: \'QWC\', \'C\' or \'AC\'')

def Build_Graph_Edges_COMMUTING_QWC_AntiCommuting(Graph, PauliWord_string_nodes_list, anti_comm_QWC, plot_graph = False):
    """

    Function builds graph edges for commuting / anticommuting / QWC PauliWords

    Args:
        PauliWord_string_nodes_list (list): list of PauliWords (str)
        Graph: networkX graph with nodes already defined
        anti_comm_QWC (str): flags to find either:
                                           qubit wise commuting (QWC) terms  -> flag = 'QWC',
                                                             commuting terms -> flag = 'C',
                                                        anti-commuting terms -> flag = 'AC'
        plot_graph (optional, bool): whether to plot graph

    Returns:
        Graph: Graph with nodes connected if they commute / QWC / anti-commute

    """

    for i in tqdm(range(len(PauliWord_string_nodes_list)), ascii=True, desc='Building Graph Edges'):

        selected_PauliWord = PauliWord_string_nodes_list[i]

        for j in range(i + 1, len(PauliWord_string_nodes_list)):
            comparison_PauliWord = PauliWord_string_nodes_list[j]

            if Commutativity(selected_PauliWord, comparison_PauliWord, anti_comm_QWC) is True:
                Graph.add_edge(PauliWord_string_nodes_list[i], PauliWord_string_nodes_list[j])
            else:
                continue

    if plot_graph == True:
        plt.figure()
        pos = nx.circular_layout(Graph)
        nx.draw(Graph, pos, with_labels=1)
        plt.show()
    return Graph

def Get_Complemenary_Graph(Graph, node_attributes_dict=None, plot_graph=False):
    Complement_Graph = nx.complement(Graph)

    if node_attributes_dict is not None:
        nx.set_node_attributes(Complement_Graph, node_attributes_dict)

    if plot_graph == True:
        plt.figure()
        pos = nx.circular_layout(Complement_Graph)
        nx.draw(Complement_Graph, pos, with_labels=1)
        plt.show()
    return Complement_Graph

def Get_clique_cover(Graph, strategy='largest_first', plot_graph=False,  node_attributes_dict=None):
    """
    https: // en.wikipedia.org / wiki / Clique_cover

    Function gets clique cover of a graph. Does this via a graph colouring approach - therefore
    strategy is important here!

    Args:
        Graph (networkx.classes.graph.Graph): networkx graph
        strategy (str): graph colouring method to find clique cover. (note is a heuristic alg)
        plot_graph (optional, bool): whether to plot graph
        node_attributes_dict (dict): Dictionary with nodes as keys and attributes as values

    Returns:
        colour_key_for_nodes (dict): A dictionary containing colours (sets) as keys and item as list of nodes
                                     that are completely connected by edges

    """
    comp_GRAPH = Get_Complemenary_Graph(Graph, node_attributes_dict=node_attributes_dict, plot_graph=False)

    greedy_colouring_output_dic = nx.greedy_color(comp_GRAPH, strategy=strategy, interchange=False)
    unique_colours = set(greedy_colouring_output_dic.values())

    colour_key_for_nodes = {}
    for colour in unique_colours:
        if node_attributes_dict is None:
            colour_key_for_nodes[colour] = [k for k in greedy_colouring_output_dic.keys()
                                            if greedy_colouring_output_dic[k] == colour]
        else:
            colour_key_for_nodes[colour] = [(k,node_attributes_dict[k]) for k in greedy_colouring_output_dic.keys()
                                            if greedy_colouring_output_dic[k] == colour]

    if plot_graph is True:
        import matplotlib.cm as cm
        colour_list = cm.rainbow(np.linspace(0, 1, len(colour_key_for_nodes)))
        pos = nx.circular_layout(Graph)

        if node_attributes_dict is None:
            for colour in colour_key_for_nodes:
                nx.draw_networkx_nodes(Graph, pos,
                                       nodelist=[node for node in colour_key_for_nodes[colour]],
                                       node_color=colour_list[colour].reshape([1,4]),
                                       node_size=500,
                                       alpha=0.8
                                       )
        else:
            for colour in colour_key_for_nodes:
                nx.draw_networkx_nodes(Graph, pos,
                                       nodelist=[node_dict_tuple[0] for node_dict_tuple in colour_key_for_nodes[colour]],
                                       node_color=colour_list[colour].reshape([1,4]),
                                       node_size=500,
                                       alpha=0.8
                                       )

        labels = {node: node for node in list(Graph.nodes)}
        nx.draw_networkx_labels(Graph, pos, labels)  # , font_size=8)

        nx.draw_networkx_edges(Graph, pos, width=1.0, alpha=0.5)
        nx.draw_networkx_edges(Graph, pos, width=1.0, alpha=0.5)

    return colour_key_for_nodes

def Get_PauliWord_constant_tuples(coloured_sets, dict_str_label='Cofactors'):
    """
    Function to give back list of PauliWords and cofactor tuples


    Args:
        coloured_sets (dict): Dictionary of PauliWords and node attributes
        dict_str_label (str): string key of Hamiltonian cofactors

    Returns:
        stripped_set (dict): dictionary of sets, each is a list of (PauliWord, constant)

    coloured_sets = {
                0: [{'I0 I1 I2 I3': {'Cofactors': (-0.32760818995565577+0j),
                                        'random_attribute': 0}}],
                1: [{'I0 Z1 I2 I3': {'Cofactors': (0.1371657293179602+0j),
                                        'random_attribute': 2}},
                    {'Y0 Y1 X2 X3': {'Cofactors': (-0.04919764587885283+0j),
                                     'random_attribute': 7}}],
                2: [{'I0 I1 Z2 I3': {'Cofactors': (-0.13036292044009176+0j),
                                        'random_attribute': 3}},
                     {'X0 X1 Y2 Y3': {'Cofactors': (-0.04919764587885283+0j),
                                        'random_attribute': 8}}],
                3: [{'I0 I1 I2 Z3': {'Cofactors': (-0.13036292044009176+0j),
                                      'random_attribute': 4}},
                     {'X0 Y1 Y2 X3': {'Cofactors': (0.04919764587885283+0j),
                                       'random_attribute': 9}}]
            }

    gives:
             0: [('I0 I1 I2 I3', (-0.32760818995565577+0j))],
             1: [('I0 Z1 I2 I3', (0.1371657293179602+0j)), ('Y0 Y1 X2 X3', (-0.04919764587885283+0j))],
             2: [('I0 I1 Z2 I3', (-0.13036292044009176+0j)), ('X0 X1 Y2 Y3', (-0.04919764587885283+0j))],
             3: [('I0 I1 I2 Z3', (-0.13036292044009176+0j)), ('X0 Y1 Y2 X3', (0.04919764587885283+0j))]}

    """
    stripped_set = {}
    for key in coloured_sets:
        temp_term_list = [(P_word[0], P_word[1][dict_str_label]) for P_word in coloured_sets[key]]
        stripped_set[key] = temp_term_list
    return stripped_set


# without using indices
if __name__ == '__main__':
    List_PauliWords = [[(0, 'I'), (1, 'I'), (2, 'I'), (3, 'I')],
     [(0, 'Z'), (1, 'I'), (2, 'I'), (3, 'I')],
     [(0, 'I'), (1, 'Z'), (2, 'I'), (3, 'I')],
     [(0, 'I'), (1, 'I'), (2, 'Z'), (3, 'I')],
     [(0, 'I'), (1, 'I'), (2, 'I'), (3, 'Z')],
     [(0, 'Z'), (1, 'Z'), (2, 'I'), (3, 'I')],
     [(0, 'Y'), (1, 'X'), (2, 'X'), (3, 'Y')],
     [(0, 'Y'), (1, 'Y'), (2, 'X'), (3, 'X')],
     [(0, 'X'), (1, 'X'), (2, 'Y'), (3, 'Y')],
     [(0, 'X'), (1, 'Y'), (2, 'Y'), (3, 'X')],
     [(0, 'Z'), (1, 'I'), (2, 'Z'), (3, 'I')],
     [(0, 'Z'), (1, 'I'), (2, 'I'), (3, 'Z')],
     [(0, 'I'), (1, 'Z'), (2,
                           'Z'), (3, 'I')],
     [(0, 'I'), (1, 'Z'), (2, 'I'), (3, 'Z')],
     [(0, 'I'), (1, 'I'), (2, 'Z'), (3, 'Z')]]
    HamiltonainCofactors = [(-0.32760818995565577 + 0j),
                            (0.1371657293179602 + 0j),
                            (0.1371657293179602 + 0j),
                            (-0.13036292044009176 + 0j),
                            (-0.13036292044009176 + 0j),
                            (0.15660062486143395 + 0j),
                            (0.04919764587885283 + 0j),
                            (-0.04919764587885283 + 0j),
                            (-0.04919764587885283 + 0j),
                            (0.04919764587885283 + 0j),
                            (0.10622904488350779 + 0j),
                            (0.15542669076236065 + 0j),
                            (0.15542669076236065 + 0j),
                            (0.10622904488350779 + 0j),
                            (0.1632676867167479 + 0j)]

    List_of_nodes = Get_PauliWords_as_nodes(List_PauliWords)
    attribute_dictionary = {'Cofactors': HamiltonainCofactors, 'random_attribute': [i for i in range(len(HamiltonainCofactors))]}

    List_of_nodes, node_attributes_dict = Get_list_of_nodes_and_attributes(List_of_nodes,
                                                                           attribute_dictionary=attribute_dictionary)

    G = nx.Graph()
    G = Build_Graph_Nodes(List_of_nodes, G, node_attributes_dict=node_attributes_dict, plot_graph=False)
    G = Build_Graph_Edges_COMMUTING_QWC_AntiCommuting(G, List_of_nodes, 'AC', plot_graph = True)

    anti_commuting_set = Get_clique_cover(G, strategy='largest_first', plot_graph=False,
                                          node_attributes_dict=node_attributes_dict)

    anti_commuting_set_stripped = Get_PauliWord_constant_tuples(anti_commuting_set, dict_str_label='Cofactors')
    print(anti_commuting_set_stripped)

    #comp_G = Get_Complemenary_Graph(G, node_attributes_dict=node_attributes_dict, plot_graph=True) # <- not currently used


#using indices
if __name__ == '__main__':
    List_PauliWords = [[(0, 'I'), (1, 'I'), (2, 'I'), (3, 'I')],
     [(0, 'Z'), (1, 'I'), (2, 'I'), (3, 'I')],
     [(0, 'I'), (1, 'Z'), (2, 'I'), (3, 'I')],
     [(0, 'I'), (1, 'I'), (2, 'Z'), (3, 'I')],
     [(0, 'I'), (1, 'I'), (2, 'I'), (3, 'Z')],
     [(0, 'Z'), (1, 'Z'), (2, 'I'), (3, 'I')],
     [(0, 'Y'), (1, 'X'), (2, 'X'), (3, 'Y')],
     [(0, 'Y'), (1, 'Y'), (2, 'X'), (3, 'X')],
     [(0, 'X'), (1, 'X'), (2, 'Y'), (3, 'Y')],
     [(0, 'X'), (1, 'Y'), (2, 'Y'), (3, 'X')],
     [(0, 'Z'), (1, 'I'), (2, 'Z'), (3, 'I')],
     [(0, 'Z'), (1, 'I'), (2, 'I'), (3, 'Z')],
     [(0, 'I'), (1, 'Z'), (2, 'Z'), (3, 'I')],
     [(0, 'I'), (1, 'Z'), (2, 'I'), (3, 'Z')],
     [(0, 'I'), (1, 'I'), (2, 'Z'), (3, 'Z')]]
    Node_and_connected_Nodes = [(0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
                 (1, [0, 2, 3, 4, 5, 10, 11, 12, 13, 14]),
                 (2, [0, 1, 3, 4, 5, 10, 11, 12, 13, 14]),
                 (3, [0, 1, 2, 4, 5, 10, 11, 12, 13, 14]),
                 (4, [0, 1, 2, 3, 5, 10, 11, 12, 13, 14]),
                 (5, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
                 (6, [0, 5, 7, 8, 9, 10, 11, 12, 13, 14]),
                 (7, [0, 5, 6, 8, 9, 10, 11, 12, 13, 14]),
                 (8, [0, 5, 6, 7, 9, 10, 11, 12, 13, 14]),
                 (9, [0, 5, 6, 7, 8, 10, 11, 12, 13, 14]),
                 (10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]),
                 (11, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14]),
                 (12, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]),
                 (13, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14]),
                 (14, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])]
    HamiltonainCofactors = [(-0.32760818995565577 + 0j),
                            (0.1371657293179602 + 0j),
                            (0.1371657293179602 + 0j),
                            (-0.13036292044009176 + 0j),
                            (-0.13036292044009176 + 0j),
                            (0.15660062486143395 + 0j),
                            (0.04919764587885283 + 0j),
                            (-0.04919764587885283 + 0j),
                            (-0.04919764587885283 + 0j),
                            (0.04919764587885283 + 0j),
                            (0.10622904488350779 + 0j),
                            (0.15542669076236065 + 0j),
                            (0.15542669076236065 + 0j),
                            (0.10622904488350779 + 0j),
                            (0.1632676867167479 + 0j)]

    List_of_nodes = Get_PauliWords_as_nodes(List_PauliWords)
    attribute_dictionary = {'Cofactors': HamiltonainCofactors, 'random_attribute': [i for i in range(len(HamiltonainCofactors))]}

    List_of_nodes, node_attributes_dict = Get_list_of_nodes_and_attributes(List_of_nodes,
                                                                           attribute_dictionary=attribute_dictionary)

    G = nx.Graph()
    G = Build_Graph_Nodes(List_of_nodes, G, node_attributes_dict=node_attributes_dict, plot_graph=False)
    G = Build_Graph_Edges_defined_by_indices(G, Node_and_connected_Nodes, plot_graph = True)

    #comp_G = Get_Complemenary_Graph(G, node_attributes_dict=node_attributes_dict, plot_graph=True) # <- not currently used

    anti_commuting_set = Get_clique_cover(G, strategy='largest_first', plot_graph=False,
                                          node_attributes_dict=node_attributes_dict)
    print(anti_commuting_set)

    anti_commuting_set_stripped = Get_PauliWord_constant_tuples(anti_commuting_set, dict_str_label='Cofactors')


class Hamiltonian_Graph():

    def __init__(self, List_of_nodes, attribute_dictionary=None):

        self.List_of_nodes = List_of_nodes
        self.attribute_dictionary = attribute_dictionary

        self.Graph = nx.Graph()
        self.anti_commuting_set = None
        self.node_attributes_dict = None

    def _Get_node_attributes_dict(self):
        List_of_nodes, node_attributes_dict = Get_list_of_nodes_and_attributes(self.List_of_nodes,
                                                                        attribute_dictionary=self.attribute_dictionary)
        self.node_attributes_dict = node_attributes_dict

    def _Build_Graph_nodes(self, plot_graph=False):

        if self.attribute_dictionary is not None:
            self._Get_node_attributes_dict()

        self.Graph = Build_Graph_Nodes(self.List_of_nodes, self.Graph, node_attributes_dict=self.node_attributes_dict,
                                       plot_graph=plot_graph)

    def _Build_Graph_edges(self, commutativity_flag, plot_graph=False):
        self.Graph = Build_Graph_Edges_COMMUTING_QWC_AntiCommuting(self.Graph, self.List_of_nodes, commutativity_flag,
                                                                   plot_graph=plot_graph)

    def _Colour_Graph(self, Graph_colouring_strategy='largest_first', plot_graph=False):
        output_sets = Get_clique_cover(self.Graph, strategy=Graph_colouring_strategy, plot_graph=plot_graph,
                         node_attributes_dict=self.node_attributes_dict)

        return output_sets

    def Get_Pauli_grouping(self, commutativity_flag, Graph_colouring_strategy='largest_first', plot_graph=False):
        self.Graph.clear()
        self._Build_Graph_nodes(plot_graph=plot_graph)
        self._Build_Graph_edges(commutativity_flag, plot_graph=plot_graph)
        output_sets = self._Colour_Graph(Graph_colouring_strategy=Graph_colouring_strategy, plot_graph=plot_graph)
        return output_sets


def Graph_of_two_sets(PauliWord_string_nodes_list_1, PauliWord_string_nodes_list_2,
                                                  anti_comm_QWC, plot_graph=False, node_attributes_dict=None):
    """

    Function builds graph edges for commuting / anticommuting / QWC PauliWords

    Args:
        PauliWord_string_nodes_list_1 (list): list of PauliWords (str) of set 1
        PauliWord_string_nodes_list_2 (list): list of PauliWords (str) of set 2
        anti_comm_QWC (str): flags to find either:
                                           qubit wise commuting (QWC) terms  -> flag = 'QWC',
                                                             commuting terms -> flag = 'C',
                                                        anti-commuting terms -> flag = 'AC'
        plot_graph (optional, bool): whether to plot graph

    Returns:
        Graph: a networkX Graph with nodes connected if they commute / QWC / anti-commute

    """
    Graph = nx.Graph()
    # Build nodes
    labels={}
    for node in [*PauliWord_string_nodes_list_1, *PauliWord_string_nodes_list_2]:
        Graph.add_node(node)
        labels[node] = node

    if node_attributes_dict is not None:
        nx.set_node_attributes(Graph, node_attributes_dict)

    # Build Edges
    edgelist=[]
    for i in tqdm(range(len(PauliWord_string_nodes_list_1)), ascii=True, desc='Building Graph Edges'):
        selected_PauliWord = PauliWord_string_nodes_list_1[i]

        for comparison_PauliWord in PauliWord_string_nodes_list_2:
            if Commutativity(selected_PauliWord, comparison_PauliWord, anti_comm_QWC) is True:
                Graph.add_edge(selected_PauliWord, comparison_PauliWord)
                edgelist.append((selected_PauliWord,comparison_PauliWord))
            else:
                continue

    if plot_graph is True:
        pos = nx.circular_layout(Graph)

        nx.draw_networkx_nodes(Graph, pos,
                               nodelist=PauliWord_string_nodes_list_1,
                               node_color='r',
                               node_size=500,
                               alpha=0.8)
        nx.draw_networkx_nodes(Graph, pos,
                               nodelist=PauliWord_string_nodes_list_2,
                               node_color='b',
                               node_size=500,
                               alpha=0.8)

        nx.draw_networkx_labels(Graph, pos, labels)  # , font_size=8)

        nx.draw_networkx_edges(Graph, pos,
                               edgelist=edgelist,
                               width=2, alpha=0.5, edge_color='k')

        plt.show()

    return Graph


def Check_if_sets_completely_connected(GRAPH, set1_P, set2_P):
    """

    Function checks if graph of two sets is completely connected or not

    Args:
        GRAPH (networkx.classes.graph.Graph): networkX graph of two sets
        set1_P (list): list of PauliWords (str) of set 1
        set2_P (list): list of PauliWords (str) of set 2

    Returns:
        Bool: True if completely connected

    """
    adj_mat = nx.adjacency_matrix(GRAPH, nodelist=[*set1_P, *set2_P])

    # select correct part of adjacency matrix!
    check_connected = adj_mat[:len(set1_P), len(set1_P):len(set1_P)+len(set2_P)]

    #Get number of connected terms
    num_non_zero = check_connected.nnz

    #Get number of connected terms if completely connected
    num_non_zero_full = check_connected.shape[0]*check_connected.shape[1]

    if num_non_zero == num_non_zero_full:
        return True
    else:
        return False


def Get_subgraph_of_coloured_graph(anti_commuting_set_stripped, anti_comm_QWC):
    """

    Function takes in a  dictionary of sets (anti_commuting_set_stripped), where each
    value is a list of (PauliWord, constant). It takes the KEY of each set and works out what other KEYS
    commute with it. NOTE this is keys of the dictionary - to be used to select terms in anti_commuting_set_stripped

    Args:
        anti_commuting_set_stripped (dict):  dictionary of sets, each is a list of (PauliWord, constant)
        anti_comm_QWC (str): flags to find either:
                                           qubit wise commuting (QWC) terms  -> flag = 'QWC',
                                                             commuting terms -> flag = 'C',
                                                        anti-commuting terms -> flag = 'AC'
    Returns:
        GRAPH_key_nodes: a networkX Graph with nodes as sets and connection edge when everything between sets is QWC, C or AC

    anti_commuting_set_stripped =
        {
         0: [('I0 I1 I2 I3 I4 I5 I6 I7 I8 I9 I10 I11 I12 I13',(-46.46560078368167+0j))],
         1: [('I0 I1 I2 I3 I4 I5 I6 I7 Z8 Z9 I10 I11 I12 I13',(0.2200397733437614+0j))],
         2: [('I0 I1 I2 I3 I4 I5 I6 I7 Z8 I9 I10 I11 I12 I13', (1.3692852793098753+0j)),
          ('Y0 X1 I2 I3 I4 I5 I6 I7 X8 Y9 I10 I11 I12 I13', (0.006509359453068324+0j)),
          ('Y0 Z1 Z2 X3 I4 I5 I6 I7 X8 Y9 I10 I11 I12 I13', (-0.00812685940290403+0j)),
          ('I0 X1 X2 I3 I4 I5 I6 I7 X8 X9 I10 I11 I12 I13', (0.00812685940290403+0j)),
          ('Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9 I10 I11 I12 I13',(-0.0033479000466714085+0j)),
          ('I0 X1 Z2 Z3 Z4 Z5 X6 I7 X8 X9 I10 I11 I12 I13',(0.0033479000466714085+0j)),
          ('Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11 I12 I13', (-0.0038801488134230276+0j)),
          ('I0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10 I11 I12 I13',(-0.0038801488134230276+0j))]
       }
    anti_comm_QWC = 'C'

    """
    GRAPH_key_nodes = nx.Graph()
    for key in anti_commuting_set_stripped:
        GRAPH_key_nodes.add_node(key)

    for key in anti_commuting_set_stripped:
        set1_P, set1_C = zip(*anti_commuting_set_stripped[key])

        for k in range(key + 1, len(anti_commuting_set_stripped)):
            set2_P, set2_C = zip(*anti_commuting_set_stripped[k])

            Graph_of_sets = Graph_of_two_sets(set1_P, set2_P,
                                              anti_comm_QWC, plot_graph=False, node_attributes_dict=None)

            if Check_if_sets_completely_connected(Graph_of_sets, set1_P, set2_P):
                GRAPH_key_nodes.add_edge(key, k)  # connection of anti_commuting set key if completely connected

    return GRAPH_key_nodes