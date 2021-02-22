import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


########### Hamiltonian Graph Functions ##############

def OpenFermion_Commutativity(QubitOp_1_frozen, QubitOp_2_frozen, Comm_flag):
    """

     Find if two PauliWords either commute or anti_commute.
     By default it will check if they commute.

    Args:
        QubitOp_1 ( openfermion.ops._qubit_operator.QubitOperator): First PauliWord to compare
        QubitOp_2 ( openfermion.ops._qubit_operator.QubitOperator): Second PauliWord to compare
        Comm_flag (str): flags to find either:
                                                   qubit wise commuting (QWC) terms  -> flag = 'QWC',
                                                                     commuting terms -> flag = 'C',
                                                                anti-commuting terms -> flag = 'AC'

    Returns:
        (bool): True or false as to whether terms commute or anti_commute

    """

    # checker = np.zeros(len(P1))

    QubitOp_1_PauliStrs, _ = QubitOp_1_frozen
    if QubitOp_1_PauliStrs:
        qubitNo_Op1, PauliStr_Op1 = list(zip(*QubitOp_1_PauliStrs))
        qubitNo_Op1 = np.array(qubitNo_Op1)
    else:
        #identity term!
        if Comm_flag == 'C':
            return True
        elif Comm_flag == 'QWC':
            return True
        elif Comm_flag == 'AC':
            return False
        else:
            raise ValueError('unknown commutation flag')

    QubitOp_2_PauliStrs, _ = QubitOp_2_frozen
    if QubitOp_2_PauliStrs:
        qubitNo_Op2, PauliStr_Op2 = list(zip(*QubitOp_2_PauliStrs))
        qubitNo_Op2 = np.array(qubitNo_Op2)
    else:
        # identity term!
        if Comm_flag == 'C':
            return True
        elif Comm_flag == 'QWC':
            return True
        elif Comm_flag == 'AC':
            return False
        else:
            raise ValueError('unknown commutation flag')

    common_qubits = np.intersect1d(qubitNo_Op1, qubitNo_Op2)

    PauliStr_Op1_common = np.take(PauliStr_Op1, np.where(np.isin(qubitNo_Op1, common_qubits) == True)).flatten()
    PauliStr_Op2_common = np.take(PauliStr_Op2, np.where(np.isin(qubitNo_Op2, common_qubits) == True)).flatten()

    commutativity_check = np.array([1 if Pauli_op1_common == PauliStr_Op2_common[index] else -1
                                    for index, Pauli_op1_common in enumerate(PauliStr_Op1_common)])

    if Comm_flag == 'QWC':
        # QWC commuting
        if bool(np.all([x == 1 for x in commutativity_check])) is True:
            return True
        else:
            return False
    else:
        if Comm_flag == 'C':
            # Commuting
            if np.prod(commutativity_check) == 1:
                return True
            else:
                return False
        elif Comm_flag == 'AC':
            # ANTI-commuting
            if np.prod(commutativity_check) == -1:
                return True
            else:
                return False
        else:
            raise KeyError('Incorrect flag used. anti_comm_QWC must be: \'QWC\', \'C\' or \'AC\'')

def Openfermion_Build_Graph_Nodes(List_of_nodes, Graph, plot_graph=False):
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
    labels={}
    node_list=[]
    for node in List_of_nodes:
        Graph.add_node(node)

        if plot_graph is True:
            node_list.append(node)

            PauliStrs, _ = node
            PauliStr_list = [''.join(map(str,[element for element in tupl[::-1]])) for tupl in PauliStrs]
            PauliWord= ' '.join(PauliStr_list)
            if PauliWord =='':
                PauliWord='I'
            labels[node] = PauliWord


    if plot_graph is True:
        plt.figure()

        pos = nx.circular_layout(Graph)

        nx.draw_networkx_nodes(Graph, pos,
                               nodelist=node_list,
                               node_color='r',
                               node_size=500,
                               alpha=0.8)

        nx.draw_networkx_labels(Graph, pos, labels)  # , font_size=8)
        # plt.savefig('G_raw', dpi=300, transparent=True, )  # edgecolor='black', facecolor='white')
        plt.show()
    return Graph

def Openfermion_Build_Graph_Edges_COMMUTING_QWC_AntiCommuting(Graph, List_of_nodes, anti_comm_QWC, plot_graph = False):
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
    node_list=[]
    labels={}

    for index, selected_PauliWord in enumerate(tqdm(List_of_nodes, ascii=True, desc='Building Graph Edges')):

        for j in range(index + 1, len(List_of_nodes)):
            comparison_PauliWord = List_of_nodes[j]

            if OpenFermion_Commutativity(selected_PauliWord, comparison_PauliWord, anti_comm_QWC) is True:
                Graph.add_edge(selected_PauliWord, comparison_PauliWord)
            else:
                continue

        if plot_graph is True:
            node_list.append(selected_PauliWord)
            PauliStrs, _ = selected_PauliWord
            PauliStr_list = [''.join(map(str, [element for element in tupl[::-1]])) for tupl in PauliStrs]
            PauliWord = ' '.join(PauliStr_list)
            if PauliWord =='':
                PauliWord='I'
            labels[selected_PauliWord] = PauliWord

    if plot_graph is True:
        plt.figure()

        pos = nx.circular_layout(Graph)

        nx.draw_networkx_nodes(Graph, pos,
                               nodelist=node_list,
                               node_color='r',
                               node_size=500,
                               alpha=0.8)

        nx.draw_networkx_labels(Graph, pos, labels)  # , font_size=8)
        nx.draw_networkx_edges(Graph, pos, width=1.0, alpha=0.5)

        plt.savefig('anti_comm_G', dpi=300, transparent=True, )  # edgecolor='black', facecolor='white')

        plt.show()

    return Graph

def Openfermion_Get_Complemenary_Graph(Graph, plot_graph=False):

    Complement_Graph = nx.complement(Graph)

    node_list=[]
    labels={}
    if plot_graph is True:
        plt.figure()
        for node in Complement_Graph.nodes:
            node_list.append(node)
            PauliStrs, _ = node
            PauliStr_list = [''.join(map(str, [element for element in tupl[::-1]])) for tupl in PauliStrs]
            PauliWord = ' '.join(PauliStr_list)
            if PauliWord =='':
                PauliWord='I'
            labels[node] = PauliWord

        pos = nx.circular_layout(Complement_Graph)

        nx.draw_networkx_nodes(Complement_Graph, pos,
                               nodelist=node_list,
                               node_color='r',
                               node_size=500,
                               alpha=0.8)

        nx.draw_networkx_labels(Complement_Graph, pos, labels)  # , font_size=8)
        nx.draw_networkx_edges(Complement_Graph, pos, width=1.0, alpha=0.5)
        plt.savefig('comp_G', dpi=300, transparent=True, )  # edgecolor='black', facecolor='white')
        plt.show()
    return Complement_Graph

def Openfermion_Get_clique_cover(Graph, strategy='largest_first', plot_graph=False):
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
    comp_GRAPH = Openfermion_Get_Complemenary_Graph(Graph, plot_graph=False)

    greedy_colouring_output_dic = nx.greedy_color(comp_GRAPH, strategy=strategy, interchange=False)
    unique_colours = set(greedy_colouring_output_dic.values())

    colour_key_for_nodes = {}
    for colour in unique_colours:
        colour_key_for_nodes[colour] = [k for k in greedy_colouring_output_dic.keys()
                                        if greedy_colouring_output_dic[k] == colour]

    if plot_graph is True:
        import matplotlib.cm as cm
        colour_list = cm.rainbow(np.linspace(0, 1, len(colour_key_for_nodes)))
        pos = nx.circular_layout(Graph)

        for colour in colour_key_for_nodes:
            nx.draw_networkx_nodes(Graph, pos,
                                   nodelist=[node for node in colour_key_for_nodes[colour]],
                                   node_color=colour_list[colour].reshape([1,4]),
                                   node_size=500,
                                   alpha=0.8)


        # labels = {node: node for node in list(Graph.nodes)}
        seperator = ' '
        labels = {node: seperator.join([tup[1] + str(tup[0]) for tup in node[0]]) for node in list(Graph.nodes)}

        nx.draw_networkx_labels(Graph, pos, labels)  # , font_size=8)

        nx.draw_networkx_edges(Graph, pos, width=1.0, alpha=0.5)

        plt.savefig('coloured_G', dpi=300, transparent=True, )  # edgecolor='black', facecolor='white')

        plt.show()

    return colour_key_for_nodes

def Convert_Clique_Cover_to_QubitOp(clique_cover_dict):
    from openfermion.ops import QubitOperator

    qubit_op_list_clique={}
    for key in clique_cover_dict:
        qubit_op_list=[]
        for PauliStr_const in clique_cover_dict[key]:
            PauliStrs, const = PauliStr_const
            Op = QubitOperator(PauliStrs, const)
            qubit_op_list.append(Op)
        qubit_op_list_clique[key] = qubit_op_list

    return qubit_op_list_clique

def Convert_Clique_Cover_to_str(clique_cover_dict):
    from openfermion.ops import QubitOperator

    qubit_op_list_clique={}
    for key in clique_cover_dict:
        qubit_op_list=[]
        for PauliStr_const in clique_cover_dict[key]:

            PauliStrs, const = PauliStr_const
            PauliStr_list = [''.join(map(str, [element for element in tupl[::-1]])) for tupl in PauliStrs]
            PauliWord = ' '.join(PauliStr_list)

            qubit_op_list.append((PauliWord, const))
        qubit_op_list_clique[key] = qubit_op_list

    return qubit_op_list_clique

class Openfermion_Hamiltonian_Graph():

    def __init__(self, QubitHamiltonian):

        self.QubitHamiltonian = QubitHamiltonian
        self.Graph = nx.Graph()

    def _Get_hashable_Hamiltonian(self):
        # networkX requires hashable object... therefore concert QubitHamiltonian to hashable form

        self.QubitHamiltonianFrozen = tuple(frozenset((PauliStr, const) for op in self.QubitHamiltonian \
                                                      for PauliStr, const in op.terms.items()))

    def _Build_Graph_nodes(self, plot_graph=False):

        self.Graph = Openfermion_Build_Graph_Nodes(self.QubitHamiltonianFrozen, self.Graph, plot_graph=plot_graph)

    def _Build_Graph_edges(self, commutativity_flag, plot_graph=False):

        self.Graph = Openfermion_Build_Graph_Edges_COMMUTING_QWC_AntiCommuting(self.Graph, self.QubitHamiltonianFrozen,
                                                                               commutativity_flag, plot_graph = plot_graph)

    def _Colour_Graph(self, Graph_colouring_strategy='largest_first', plot_graph=False):

        output_sets = Openfermion_Get_clique_cover(self.Graph, strategy=Graph_colouring_strategy, plot_graph=plot_graph)

        return output_sets

    def Get_Clique_Cover_as_QubitOp(self, commutativity_flag, Graph_colouring_strategy='largest_first', plot_graph=False):
        self.Graph.clear()
        self._Get_hashable_Hamiltonian()
        self._Build_Graph_nodes(plot_graph=plot_graph)
        self._Build_Graph_edges(commutativity_flag, plot_graph=plot_graph)
        output_sets = self._Colour_Graph(Graph_colouring_strategy=Graph_colouring_strategy, plot_graph=plot_graph)
        qubitOperator_list = Convert_Clique_Cover_to_QubitOp(output_sets)
        return qubitOperator_list

    def Get_Clique_Cover_as_Pauli_strings(self, commutativity_flag, Graph_colouring_strategy='largest_first', plot_graph=False):
        self.Graph.clear()
        self._Get_hashable_Hamiltonian()
        self._Build_Graph_nodes(plot_graph=plot_graph)
        self._Build_Graph_edges(commutativity_flag, plot_graph=plot_graph)
        output_sets = self._Colour_Graph(Graph_colouring_strategy=Graph_colouring_strategy, plot_graph=plot_graph)
        qubitOperator_list_str = Convert_Clique_Cover_to_str(output_sets)
        return qubitOperator_list_str


####### sub graph

def Graph_of_two_sets_OLDFUNCT_NO_COLOUR(qubitOperator_list_1, qubitOperator_list_2, anti_comm_QWC, plot_graph=False):
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

    combined_sets = qubitOperator_list_1 + qubitOperator_list_2

    qubitOperator_list_combined_frozen = tuple(frozenset((PauliStr, const) for op in combined_sets \
                                             for PauliStr, const in op.terms.items()))

    Graph = nx.Graph()

    ## build nodes
    for node in qubitOperator_list_combined_frozen:
        Graph.add_node(node)

    ## build edges
    node_list = []
    labels = {}
    for index, selected_PauliWord in enumerate(tqdm(list(Graph.nodes), ascii=True, desc='Building Graph Edges')):

        for j in range(index + 1, len(Graph.nodes)):
            comparison_PauliWord = list(Graph.nodes)[j]

            if OpenFermion_Commutativity(selected_PauliWord, comparison_PauliWord, anti_comm_QWC) is True:
                Graph.add_edge(selected_PauliWord, comparison_PauliWord)
            else:
                continue

        if plot_graph is True:
            node_list.append(selected_PauliWord)
            PauliStrs, _ = selected_PauliWord
            PauliStr_list = [''.join(map(str, [element for element in tupl[::-1]])) for tupl in PauliStrs]
            PauliWord = ' '.join(PauliStr_list)
            labels[selected_PauliWord] = PauliWord

    if plot_graph is True:
        plt.figure()

        pos = nx.circular_layout(Graph)

        nx.draw_networkx_nodes(Graph, pos,
                               nodelist=node_list,
                               node_color='r',
                               node_size=500,
                               alpha=0.8)

        nx.draw_networkx_labels(Graph, pos, labels)  # , font_size=8)
        nx.draw_networkx_edges(Graph, pos, width=1.0, alpha=0.5)
        plt.show()

    return Graph

def Graph_of_two_sets(qubitOperator_list_1, qubitOperator_list_2, anti_comm_QWC, plot_graph=False):
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


    qubitOperator_list_1_terms = tuple([tuple(frozenset((PauliStr, const))) for op in qubitOperator_list_1 for
                                        PauliStr, const in op.terms.items()])
    qubitOperator_list_2_terms = tuple([tuple(frozenset((PauliStr, const))) for op in qubitOperator_list_2 for
                                        PauliStr, const in op.terms.items()])

    qubitOperator_list_combined_frozen = []
    for const, qubit_tuple in qubitOperator_list_1_terms + qubitOperator_list_2_terms:
        if isinstance(qubit_tuple, tuple):
            qubitOperator_list_combined_frozen.append((qubit_tuple, const))
        else:
            qubitOperator_list_combined_frozen.append((const, qubit_tuple))

    Graph = nx.Graph()
    ## build nodes

    for node in qubitOperator_list_combined_frozen:
        Graph.add_node(node)

    ## build edges
    node_list_1 = []
    node_list_2 = []
    labels = {}
    for index, selected_PauliWord in enumerate(tqdm(list(Graph.nodes), ascii=True, desc='Building Graph Edges')):

        for j in range(index + 1, len(Graph.nodes)):
            comparison_PauliWord = list(Graph.nodes)[j]

            if OpenFermion_Commutativity(selected_PauliWord, comparison_PauliWord, anti_comm_QWC) is True:
                Graph.add_edge(selected_PauliWord, comparison_PauliWord)
            else:
                continue

        if plot_graph is True:
            if index < len(qubitOperator_list_1_terms):
                node_list_1.append(selected_PauliWord)
            else:
                node_list_2.append(selected_PauliWord)

            PauliStrs, _ = selected_PauliWord
            PauliStr_list = [''.join(map(str, [element for element in tupl[::-1]])) for tupl in PauliStrs]
            PauliWord = ' '.join(PauliStr_list)
            labels[selected_PauliWord] = PauliWord

    if plot_graph is True:
        plt.figure()

        pos = nx.circular_layout(Graph)

        nx.draw_networkx_nodes(Graph, pos,
                               nodelist=node_list_1,
                               node_color='r',
                               node_size=500,
                               alpha=0.8)

        nx.draw_networkx_nodes(Graph, pos,
                               nodelist=node_list_2,
                               node_color='b',
                               node_size=500,
                               alpha=0.8)

        nx.draw_networkx_labels(Graph, pos, labels)  # , font_size=8)
        nx.draw_networkx_edges(Graph, pos, width=1.0, alpha=0.5)
        plt.show()

    return Graph

def Graph_of_two_sets_CONNECTED_OLD(qubitOperator_list_1, qubitOperator_list_2, anti_comm_QWC, plot_graph=False):
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

    qubitOperator_list_1_terms = tuple([tuple(frozenset((PauliStr, const))) for op in qubitOperator_list_1 for
                                        PauliStr, const in op.terms.items()])
    qubitOperator_list_2_terms = tuple([tuple(frozenset((PauliStr, const))) for op in qubitOperator_list_2 for
                                        PauliStr, const in op.terms.items()])

    qubitOperator_list_combined_frozen = []
    for const, qubit_tuple in qubitOperator_list_1_terms + qubitOperator_list_2_terms:
        if isinstance(qubit_tuple, tuple):
            qubitOperator_list_combined_frozen.append((qubit_tuple, const))
        else:
            qubitOperator_list_combined_frozen.append((const, qubit_tuple))

    Graph = nx.Graph()
    ## build nodes

    for node in qubitOperator_list_combined_frozen:
        Graph.add_node(node)

    ## build edges
    node_list_1 = []
    node_list_2 = []
    labels = {}

    # <-- NEW

    for index, selected_PauliWord in enumerate(tqdm(list(Graph.nodes), ascii=True, desc='Building Graph Edges')):

        completely_connected_check = []  # <-- NEW

        for j in range(index + 1, len(Graph.nodes)):
            comparison_PauliWord = list(Graph.nodes)[j]

            if OpenFermion_Commutativity(selected_PauliWord, comparison_PauliWord, anti_comm_QWC) is True:
                Graph.add_edge(selected_PauliWord, comparison_PauliWord)
                completely_connected_check.append(1)  # <-- NEW
            else:
                completely_connected_check.append(0)  # <-- NEW
                continue

        if plot_graph is True:
            if index < len(qubitOperator_list_1_terms):
                node_list_1.append(selected_PauliWord)
            else:
                node_list_2.append(selected_PauliWord)

            PauliStrs, _ = selected_PauliWord
            PauliStr_list = [''.join(map(str, [element for element in tupl[::-1]])) for tupl in PauliStrs]
            PauliWord = ' '.join(PauliStr_list)
            labels[selected_PauliWord] = PauliWord

    if plot_graph is True:
        plt.figure()

        pos = nx.circular_layout(Graph)

        nx.draw_networkx_nodes(Graph, pos,
                               nodelist=node_list_1,
                               node_color='r',
                               node_size=500,
                               alpha=0.8)

        nx.draw_networkx_nodes(Graph, pos,
                               nodelist=node_list_2,
                               node_color='b',
                               node_size=500,
                               alpha=0.8)

        nx.draw_networkx_labels(Graph, pos, labels)  # , font_size=8)
        nx.draw_networkx_edges(Graph, pos, width=1.0, alpha=0.5)
        plt.show()

    if completely_connected_check ==[]:
        return Graph, False
    else:
        return Graph, bool(np.prod(completely_connected_check))  # <-- NEW (True if completely connected)

def Graph_of_two_sets_CONNECTED(qubitOperator_list_1, qubitOperator_list_2, anti_comm_QWC, plot_graph=False):
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

    qubitOperator_list_1_terms = []
    frozen_1 = [tuple(frozenset((PauliStr, const))) for op in qubitOperator_list_1 for PauliStr, const in
                op.terms.items()]
    for const, qubit_tuple in frozen_1:
        if isinstance(qubit_tuple, tuple):
            qubitOperator_list_1_terms.append((qubit_tuple, const))
        else:
            qubitOperator_list_1_terms.append((const, qubit_tuple))

    qubitOperator_list_2_terms = []
    frozen_2 = [tuple(frozenset((PauliStr, const))) for op in qubitOperator_list_2 for PauliStr, const in
                op.terms.items()]
    for const, qubit_tuple in frozen_2:
        if isinstance(qubit_tuple, tuple):
            qubitOperator_list_2_terms.append((qubit_tuple, const))
        else:
            qubitOperator_list_2_terms.append((const, qubit_tuple))

    Graph = nx.Graph()

    ## build edges
    node_list_1 = []
    node_list_2 = []
    labels = {}

    completely_connected_check = []  # <-- NEW

    for index_1, selected_PauliWord in enumerate(
            tqdm(qubitOperator_list_1_terms, ascii=True, desc='Building Graph Edges')):
        Graph.add_node(selected_PauliWord)

        for index_2, comparison_PauliWord in enumerate(qubitOperator_list_2_terms):

            Graph.add_node(comparison_PauliWord)

            if OpenFermion_Commutativity(selected_PauliWord, comparison_PauliWord, anti_comm_QWC) is True:
                Graph.add_edge(selected_PauliWord, comparison_PauliWord)
                completely_connected_check.append(1)  # <-- NEW
            else:
                completely_connected_check.append(0)  # <-- NEW
                continue

        if plot_graph is True:
            node_list_1.append(selected_PauliWord)
            node_list_2.append(comparison_PauliWord)

            PauliStrs, _ = selected_PauliWord
            PauliStr_list = [''.join(map(str, [element for element in tupl[::-1]])) for tupl in PauliStrs]
            PauliWord = ' '.join(PauliStr_list)
            labels[selected_PauliWord] = PauliWord

            PauliStrs2, _ = comparison_PauliWord
            PauliStr_list2 = [''.join(map(str, [element for element in tupl[::-1]])) for tupl in PauliStrs2]
            PauliWord2 = ' '.join(PauliStr_list2)
            labels[comparison_PauliWord] = PauliWord2

    if plot_graph is True:
        plt.figure()

        pos = nx.circular_layout(Graph)

        nx.draw_networkx_nodes(Graph, pos,
                               nodelist=node_list_1,
                               node_color='r',
                               node_size=500,
                               alpha=0.8)

        nx.draw_networkx_nodes(Graph, pos,
                               nodelist=node_list_2,
                               node_color='b',
                               node_size=500,
                               alpha=0.8)

        nx.draw_networkx_labels(Graph, pos, labels)  # , font_size=8)
        nx.draw_networkx_edges(Graph, pos, width=1.0, alpha=0.5)
        plt.show()

    if completely_connected_check == []:
        return Graph, False
    else:
        return Graph, bool(np.prod(completely_connected_check))  # <-- NEW (True if completely connected)

def Get_subgraph_of_sets(set_dict_lists, anti_comm_QWC, plot_graph=False, plot_sub_graphs=False):
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

    keys_of_single_QubitOp=[]
    for key in set_dict_lists:
        qubitOp_list = set_dict_lists[key]
        if len(qubitOp_list)>1:
            GRAPH_key_nodes.add_node(key)
        else:
            keys_of_single_QubitOp.append(key)

    for key in set_dict_lists:
        qubitOperator_list_1 = set_dict_lists[key]

        if len(qubitOperator_list_1) > 1:

            for k in range(key + 1, len(set_dict_lists)):
                qubitOperator_list_2 = set_dict_lists[k]
                if len(qubitOperator_list_2) > 1:
                    G, BOOL = Graph_of_two_sets_CONNECTED(qubitOperator_list_1,
                                                        qubitOperator_list_2,
                                                        anti_comm_QWC, plot_graph=plot_sub_graphs)
                    if BOOL is True:
                        GRAPH_key_nodes.add_edge(key, k)  # connection of SET KEYS if sets completely connected

    if plot_graph is True:
        plt.figure()
        nx.draw(GRAPH_key_nodes, with_labels=1)
        plt.show()

    # # related sets
    # print(GRAPH_key_nodes.edges)

    return GRAPH_key_nodes, keys_of_single_QubitOp

class Hamiltonian_Graph_reduction(Openfermion_Hamiltonian_Graph):
    def __init__(self, QubitHamiltonian, commutativity_flag_relating_QubitHam, Graph_colouring_strategy='largest_first',
                 plot_qubit_Hamiltonian_relation_graph=False):
        super().__init__(QubitHamiltonian)

        self.Hamiltonian_QubitOp_sets = self.Get_Clique_Cover_as_QubitOp(commutativity_flag_relating_QubitHam, Graph_colouring_strategy=Graph_colouring_strategy,
                                         plot_graph=plot_qubit_Hamiltonian_relation_graph)

    def _Get_graph_of_subgraphs(self, sub_graph_relationship_C_QWC_AC, plot_graph=False):
        Graph_of_subgraph_set_keys, keys_of_single_QubitOps = Get_subgraph_of_sets(self.Hamiltonian_QubitOp_sets,
                                                                            sub_graph_relationship_C_QWC_AC, plot_graph=plot_graph,
                                                                            plot_sub_graphs=False)
        return Graph_of_subgraph_set_keys, keys_of_single_QubitOps

    def Get_Coloured_graph_of_subgraphs(self, sub_graph_relationship_C_QWC_AC, plot_graphof_subgraphs=False,
                                        info_print=True):

        Graph_of_subgraph_set_keys, keys_of_single_QubitOps =  self._Get_graph_of_subgraphs(sub_graph_relationship_C_QWC_AC, plot_graph=plot_graphof_subgraphs)

        Coloured_graph_of_subgraph_set_keys = Openfermion_Get_clique_cover(Graph_of_subgraph_set_keys)

        if info_print:
            print('No of terms in Hamiltonian reduced from {} to {} (reduction={})'.format(
                len(self.QubitHamiltonianFrozen),
                len(self.Hamiltonian_QubitOp_sets),
                len(self.QubitHamiltonianFrozen) - len(self.Hamiltonian_QubitOp_sets)))

            if sub_graph_relationship_C_QWC_AC == 'C':
                print('Number of completely commuting sets (non-unique) in reduced Hamiltonian: ',
                      len(Graph_of_subgraph_set_keys.edges))
                print('Number of completely UNIQUE commuting sets in reduced Hamiltonian= {}'.format(
                    len([Coloured_graph_of_subgraph_set_keys[key] for key in Coloured_graph_of_subgraph_set_keys if len(Coloured_graph_of_subgraph_set_keys[key]) > 1])))
                print('THESE can be measured SIMULTANEOUSLY')

            elif sub_graph_relationship_C_QWC_AC == 'AC':
                print('Number of completely ANTI commuting sets (non-unique) in reduced Hamiltonian: ',
                      len(Graph_of_subgraph_set_keys.edges))
                print('Number of completely UNIQUE ANTI commuting sets in reduced Hamiltonian= {}'.format(
                    len([Coloured_graph_of_subgraph_set_keys[key] for key in Coloured_graph_of_subgraph_set_keys if len(Coloured_graph_of_subgraph_set_keys[key]) > 1])))
                print('THESE can be measured SIMULTANEOUSLY')
            else:
                print('Number of completely qubit wise commuting sets (non-unique) in reduced Hamiltonian: ',
                      len(Graph_of_subgraph_set_keys.edges))
                print('Number of completely UNIQUE qubit wise commuting sets in reduced Hamiltonian= {}'.format(
                    len([Coloured_graph_of_subgraph_set_keys[key] for key in Coloured_graph_of_subgraph_set_keys if len(Coloured_graph_of_subgraph_set_keys[key]) > 1])))
                print('THESE can be measured SIMULTANEOUSLY')

            print('No of terms in reduced Hamiltonian reduced = {} BUT only requires to {} measurements'.format(
                len(self.Hamiltonian_QubitOp_sets),
                len(Coloured_graph_of_subgraph_set_keys) + len(keys_of_single_QubitOps)))

        return Coloured_graph_of_subgraph_set_keys, keys_of_single_QubitOps



# taken_indices=[]
# unqiue_terms =[]
# for a,b in GG.edges:
#     if a not in taken_indices and b not in taken_indices:
#         unqiue_terms.append((a,b))
#         taken_indices.append(a)
#         taken_indices.append(b)
# print(len(unqiue_terms))

# qubitOperator_list_1= AC_set_dict_list[35]
# qubitOperator_list_2 = AC_set_dict_list[44]
# anti_comm_QWC='C'
# Check_if_graph_of_two_sets_completely_connected(qubitOperator_list_1, qubitOperator_list_2, anti_comm_QWC, plot_graph=False)