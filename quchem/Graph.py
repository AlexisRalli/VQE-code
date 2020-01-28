import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
from tqdm import tqdm


class BuildGraph_string():
    """

    The BuildGraph string object calculates and retains all the unitary coupled cluster terms.

    Args:
        PauliWords (list): List of tuples as (qubitNo (int), PauliString (str)) of Hamiltonian
        indices (list): List of Tuples as (Index of PauliWord, list_of_indices PauliWord commutes with)
        HamiltonianCofactors (list): List of Hamiltonian constants. Should match order of PauliWords arg.

    Attributes:

        PauliWords (list): List of tuples as (qubitNo (int), PauliString (str)) of Hamiltonian
        indices (list): List of Tuples as (Index of PauliWord, list_of_indices PauliWord commutes with)
        HamiltonianCofactors (list): List of Hamiltonian constants. Should match order of PauliWords arg.


        G_string (networkx.classes.graph.Graph): NetworkX graph - where each node is PauliWord and connected by edge
                                                 if node (another PauliWord) commutes with it.

        node_string_set (list): List of PauliWords (str)

        node_string_set_and_HamiltonianCofactors (dict): Dictionary of PauliWord (key) and HamiltonianCofactors (value).
                                                         Important to set node attributes.

        G_string_comp (networkx.classes.graph.Graph): Complementary graph of G_string. Nodes now connected if they
                                                      anti-commute

        greedy_string (list):
        colour_key_for_nodes_string (list):

        anticommuting_sets (list):

    """
    def __init__(self, PauliWords, indices, HamiltonainCofactors):

        self.PauliWords = PauliWords
        self.indices = indices
        self.HamiltonainCofactors = HamiltonainCofactors


        self.G_string = nx.Graph() # undirected graph
        self.node_string_set = None
        self.node_string_set_and_HamiltonainCofactors = None
        self.G_string_comp = None
        self.greedy_string = None
        self.colour_key_for_nodes_string = None
        self.max_clique_cover=[]
        self.anticommuting_sets = None


    def Get_node_terms_as_strings(self):

        node_string_set = []
        node_string_set_and_HamiltonainCofactors = {}

        for index, commuting_indices in self.indices:
            PauliWord = self.PauliWords[index]
            Cofactor = self.HamiltonainCofactors[index]

            PauliStrings = ['{}{}'.format(qubitOp, qubitNo) for qubitNo, qubitOp in PauliWord]

            seperator = ' '
            node_string_set.append(seperator.join(PauliStrings))

            node_string_set_and_HamiltonainCofactors.update({seperator.join(PauliStrings): Cofactor})

        self.node_string_set = node_string_set
        self.node_string_set_and_HamiltonainCofactors = node_string_set_and_HamiltonainCofactors




    def Build_string_nodes(self, plot_graph = False):

        if self.node_string_set is None:
            self.Get_node_terms_as_strings()

        for string_node in self.node_string_set:
            self.G_string.add_node(string_node)


        if plot_graph == True:
            plt.figure()
            nx.draw(self.G_string, with_labels=1)
            plt.show()

        # add nodes attributes (appends Hamiltonian cofactors to each node)
        # access via: self.G_string.nodes[NODE-NAME]['Cofactor']
        # e.g. X.G_string.nodes['I0 I1 I2 I3']['Cofactor']
        nx.set_node_attributes(self.G_string, self.node_string_set_and_HamiltonainCofactors, 'Cofactor')



    def Build_string_edges(self, plot_graph = False):

        if len(self.G_string.nodes()) == 0:
            self.Build_string_nodes()

        nodes_list = list(self.G_string.nodes())
        for index, commuting_indices in self.indices:
            for commuting_index in commuting_indices:
                if commuting_index != []:
                    self.G_string.add_edge(nodes_list[index], nodes_list[commuting_index])

        if plot_graph == True:
            plt.figure()
            pos = nx.circular_layout(self.G_string)
            nx.draw(self.G_string, pos, with_labels=1)
            plt.show()


    def Get_complementary_graph_string(self, plot_graph = False):

        if len(list(self.G_string.edges())) == 0:
            self.Build_string_edges()

        self.G_string_comp = nx.complement(self.G_string)

        nx.set_node_attributes(self.G_string_comp, self.node_string_set_and_HamiltonainCofactors, 'Cofactor')

        if plot_graph == True:
            plt.figure()
            pos = nx.circular_layout(self.G_string_comp)
            nx.draw(self.G_string_comp, pos, with_labels=1)
            plt.show()


    def _Get_subgraphs(self):

        # -*- coding: utf-8 -*-
        #    Copyright (C) 2004-2016 by
        #    Aric Hagberg <hagberg@lanl.gov>
        #    Dan Schult <dschult@colgate.edu>
        #    Pieter Swart <swart@lanl.gov>
        #    All rights reserved.
        #    BSD license.
        #
        # Authors: Eben Kenah
        #          Aric Hagberg (hagberg@lanl.gov)
        #          Christopher Ellison
        """Connected components."""
        import networkx as nx
        from networkx.utils.decorators import not_implemented_for

        __all__ = [
            'number_connected_components',
            'connected_components',
            'connected_component_subgraphs',
            'is_connected',
            'node_connected_component',
        ]

        @not_implemented_for('directed')
        def connected_components(G):
            """Generate connected components.

            Parameters
            ----------
            G : NetworkX graph
               An undirected graph

            Returns
            -------
            comp : generator of sets
               A generator of sets of nodes, one for each component of G.

            Raises
            ------
            NetworkXNotImplemented:
                If G is undirected.

            Examples
            --------
            Generate a sorted list of connected components, largest first.

            # >>> G = nx.path_graph(4)
            # >>> nx.add_path(G, [10, 11, 12])
            # >>> [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
            [4, 3]

            If you only want the largest connected component, it's more
            efficient to use max instead of sort.

            # >>> largest_cc = max(nx.connected_components(G), key=len)

            See Also
            --------
            strongly_connected_components
            weakly_connected_components

            Notes
            -----
            For undirected graphs only.

            """
            seen = set()
            for v in G:
                if v not in seen:
                    c = set(_plain_bfs(G, v))
                    yield c
                    seen.update(c)
        @not_implemented_for('directed')
        def connected_component_subgraphs(G, copy=True):
            """Generate connected components as subgraphs.

            Parameters
            ----------
            G : NetworkX graph
               An undirected graph.

            copy: bool (default=True)
              If True make a copy of the graph attributes

            Returns
            -------
            comp : generator
              A generator of graphs, one for each connected component of G.

            Raises
            ------
            NetworkXNotImplemented:
                If G is undirected.

            Examples
            --------
            # >>> G = nx.path_graph(4)
            # >>> G.add_edge(5,6)
            # >>> graphs = list(nx.connected_component_subgraphs(G))

            If you only want the largest connected component, it's more
            efficient to use max instead of sort:

            # >>> Gc = max(nx.connected_component_subgraphs(G), key=len)

            See Also
            --------
            connected_components
            strongly_connected_component_subgraphs
            weakly_connected_component_subgraphs

            Notes
            -----
            For undirected graphs only.
            Graph, node, and edge attributes are copied to the subgraphs by default.

            """
            for c in connected_components(G):
                if copy:
                    yield G.subgraph(c).copy()
                else:
                    yield G.subgraph(c)
        @not_implemented_for('directed')
        def is_connected(G):
            """Return True if the graph is connected, false otherwise.

            Parameters
            ----------
            G : NetworkX Graph
               An undirected graph.

            Returns
            -------
            connected : bool
              True if the graph is connected, false otherwise.

            Raises
            ------
            NetworkXNotImplemented:
                If G is undirected.

            Examples
            --------
            # >>> G = nx.path_graph(4)
            # >>> print(nx.is_connected(G))
            True

            See Also
            --------
            is_strongly_connected
            is_weakly_connected
            is_semiconnected
            is_biconnected
            connected_components

            Notes
            -----
            For undirected graphs only.

            """
            if len(G) == 0:
                raise nx.NetworkXPointlessConcept('Connectivity is undefined ',
                                                  'for the null graph.')
            return len(set(_plain_bfs(G, arbitrary_element(G)))) == len(G)
        @not_implemented_for('directed')
        def node_connected_component(G, n):
            """Return the nodes in the component of graph containing node n.

            Parameters
            ----------
            G : NetworkX Graph
               An undirected graph.

            n : node label
               A node in G

            Returns
            -------
            comp : set
               A set of nodes in the component of G containing node n.

            Raises
            ------
            NetworkXNotImplemented:
                If G is directed.

            See Also
            --------
            connected_components

            Notes
            -----
            For undirected graphs only.

            """
            return set(_plain_bfs(G, n))
        def _plain_bfs(G, source):
            """A fast BFS node generator"""
            seen = set()
            nextlevel = {source}
            while nextlevel:
                thislevel = nextlevel
                nextlevel = set()
                for v in thislevel:
                    if v not in seen:
                        yield v
                        seen.add(v)
                        nextlevel.update(G[v])

        connected_graphs = list(connected_component_subgraphs(self.G_string))
        multi_node_G = []
        single_node_G = []
        for graph in connected_graphs:

            if len(graph.nodes) > 1:
                # pos = nx.circular_layout(graph)
                # plt.figure()
                # nx.draw(graph, pos, with_labels=1)
                nx.set_node_attributes(graph, self.node_string_set_and_HamiltonainCofactors, 'Cofactor')
                multi_node_G.append(graph)
            else:
                nx.set_node_attributes(graph, self.node_string_set_and_HamiltonainCofactors, 'Cofactor')
                single_node_G.append(graph)

        #self.connected_graphs = connected_graphs
        return single_node_G, multi_node_G

    def colouring(self, plot_graph = False, strategy='largest_first'):
        # different strategies at:
        # https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.coloring.greedy_color.html#networkx.algorithms.coloring.greedy_color

        single_node_G, multi_node_G = self._Get_subgraphs()


        multi_node_G_coloured = []
        for graph in multi_node_G:
            greedy_string = nx.greedy_color(graph, strategy=strategy, interchange=False)

            unique_colours = set(greedy_string.values())

            colour_key_for_nodes_string = {}
            for colour in unique_colours:
                # colour_key_for_nodes_string[colour] = [k for k in greedy_string.keys()
                #                                        if greedy_string[k] == colour]

                colour_key_for_nodes_string[colour] = [(k, graph.nodes[k]['Cofactor']) for k in
                                                       greedy_string.keys()
                                                       if greedy_string[k] == colour]

            multi_node_G_coloured.append(colour_key_for_nodes_string)

            if plot_graph == True:
                import matplotlib.cm as cm

                plt.figure()
                colour_list = cm.rainbow(np.linspace(0, 1, len(colour_key_for_nodes_string)))
                pos = nx.circular_layout(graph)

                for colour in colour_key_for_nodes_string:
                    nx.draw_networkx_nodes(graph, pos,
                                           nodelist=[PauliWord for PauliWord, const in colour_key_for_nodes_string[colour]],
                                           node_color=colour_list[colour],
                                           node_size=500,
                                           alpha=0.8)

                nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)


                # available fonts!
                # import matplotlib
                # avail_font_names = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
                nx.draw_networkx_labels(graph, pos, font_family='Mitra Mono', font_size=12)
                plt.plot()


        single_node_G_coloured = []

        for comp_graph in single_node_G:
            greedy_string = nx.greedy_color(comp_graph, strategy=strategy, interchange=False)

            Cofactor = comp_graph.nodes[list(comp_graph.nodes)[0]]['Cofactor']

            single_node_G_coloured.append(dict([(value, (key, Cofactor)) for key, value in greedy_string.items()]))



        #self.single_node_G_coloured = single_node_G_coloured
        #self.multi_node_G_coloured = multi_node_G_coloured


        iter = 0
        anti_commuting_sets_dict = {}
        for sub_graph in single_node_G_coloured + multi_node_G_coloured:
            for key, value in sub_graph.items():
                anti_commuting_sets_dict[iter] = value
                iter += 1

        self.anticommuting_sets = anti_commuting_sets_dict

    # def max_clique_cover_composite_graph(self):
    #
    #     cliques = list(nx.find_cliques(self.G_string_comp))
    #     sorted_cliques = sorted(cliques, key=len, reverse=True)
    #     clique_list = []
    #
    #     #for clique in sorted_cliques:
    #     for j in tqdm(range(len(sorted_cliques)), ascii=True, desc='Getting anti-commuting sets'):
    #         clique = sorted_cliques[j]
    #         if clique_list == []:
    #             clique_list.append(clique)
    #         else:
    #             checker = [i for i in clique for cc in clique_list if i in cc]
    #             if len(checker) > 0:
    #                 # checks if have any duplicate nodes... if so then continue
    #                 continue
    #             else:
    #                 clique_list.append(clique)
    #
    #
    #     print(clique_list)
    #     self.anticommuting_sets = clique_list
    #
    #     max_clique_cover_cofactors = nx.get_node_attributes(self.G_string_comp, 'Cofactor')
    #
    #     for SET in clique_list:
    #         temp_list=[]
    #         for PauliWord in SET:
    #             temp_list.append((max_clique_cover_cofactors[PauliWord], PauliWord))
    #         self.max_clique_cover.append(temp_list)
    #
    #     # TODO maybe draw graph of max_clique

if __name__ == '__main__':
    PauliWords = [[(0, 'I'), (1, 'I'), (2, 'I'), (3, 'I')],
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
    indices = [(0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
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

    X = BuildGraph_string(PauliWords, indices, HamiltonainCofactors)

    X.Build_string_nodes()  # plot_graph=True)
    X.Build_string_edges()  # plot_graph=True)
    X.Get_complementary_graph_string()  # plot_graph=True)
    X.colouring(plot_graph=True)
    print(X.anticommuting_sets)



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

        j_list = [j for j in range(len(PauliWord_string_nodes_list)) if j != i] # all the other indexes

        for j in j_list:
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

def Get_subgraphs(Graph, node_attributes_dict=None):

    # -*- coding: utf-8 -*-
    #    Copyright (C) 2004-2016 by
    #    Aric Hagberg <hagberg@lanl.gov>
    #    Dan Schult <dschult@colgate.edu>
    #    Pieter Swart <swart@lanl.gov>
    #    All rights reserved.
    #    BSD license.
    #
    # Authors: Eben Kenah
    #          Aric Hagberg (hagberg@lanl.gov)
    #          Christopher Ellison
    """Connected components."""
    import networkx as nx


    if isinstance(Graph, nx.classes.digraph.DiGraph):
        raise TypeError('Cannot have a directed graph, must be a undirected graph')


    def connected_components(G):
        """Generate connected components.

        Parameters
        ----------
        G : NetworkX graph
           An undirected graph

        Returns
        -------
        comp : generator of sets
           A generator of sets of nodes, one for each component of G.

        Raises
        ------
        NetworkXNotImplemented:
            If G is undirected.

        Examples
        --------
        Generate a sorted list of connected components, largest first.

        # >>> G = nx.path_graph(4)
        # >>> nx.add_path(G, [10, 11, 12])
        # >>> [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
        [4, 3]

        If you only want the largest connected component, it's more
        efficient to use max instead of sort.

        # >>> largest_cc = max(nx.connected_components(G), key=len)

        See Also
        --------
        strongly_connected_components
        weakly_connected_components

        Notes
        -----
        For undirected graphs only.

        """
        seen = set()
        for v in G:
            if v not in seen:
                c = set(_plain_bfs(G, v))
                yield c
                seen.update(c)
    def connected_component_subgraphs(G, copy=True):
        """Generate connected components as subgraphs.

        Parameters
        ----------
        G : NetworkX graph
           An undirected graph.

        copy: bool (default=True)
          If True make a copy of the graph attributes

        Returns
        -------
        comp : generator
          A generator of graphs, one for each connected component of G.

        Raises
        ------
        NetworkXNotImplemented:
            If G is undirected.

        Examples
        --------
        # >>> G = nx.path_graph(4)
        # >>> G.add_edge(5,6)
        # >>> graphs = list(nx.connected_component_subgraphs(G))

        If you only want the largest connected component, it's more
        efficient to use max instead of sort:

        # >>> Gc = max(nx.connected_component_subgraphs(G), key=len)

        See Also
        --------
        connected_components
        strongly_connected_component_subgraphs
        weakly_connected_component_subgraphs

        Notes
        -----
        For undirected graphs only.
        Graph, node, and edge attributes are copied to the subgraphs by default.

        """
        for c in connected_components(G):
            if copy:
                yield G.subgraph(c).copy()
            else:
                yield G.subgraph(c)
    def _plain_bfs(G, source):
        """A fast BFS node generator"""
        seen = set()
        nextlevel = {source}
        while nextlevel:
            thislevel = nextlevel
            nextlevel = set()
            for v in thislevel:
                if v not in seen:
                    yield v
                    seen.add(v)
                    nextlevel.update(G[v])

    connected_graphs = list(connected_component_subgraphs(Graph))
    multi_node_G = []
    single_node_G = []
    for graph in connected_graphs:
        if len(graph.nodes) > 1:
            # pos = nx.circular_layout(graph)
            # plt.figure()
            # nx.draw(graph, pos, with_labels=1)
            if node_attributes_dict is not None:
                nx.set_node_attributes(graph, node_attributes_dict)
            multi_node_G.append(graph)
        else:
            if node_attributes_dict is not None:
                nx.set_node_attributes(graph, node_attributes_dict)
            single_node_G.append(graph)

    return single_node_G, multi_node_G

def Colour_list_of_Graph(Graph_list, attribute_dictionary=None, plot_graph=False, strategy='largest_first'):
    # different strategies at:
    # https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.coloring.greedy_color.html#networkx.algorithms.coloring.greedy_color

    List_of_Coloured_Graphs_dicts = []
    for graph in Graph_list:
        greedy_colouring_output_dic = nx.greedy_color(graph, strategy=strategy, interchange=False)

        unique_colours = set(greedy_colouring_output_dic.values())

        colour_key_for_nodes = {}
        for colour in unique_colours:
            if attribute_dictionary is None:
                colour_key_for_nodes[colour] = [k for k in greedy_colouring_output_dic.keys()
                                                       if greedy_colouring_output_dic[k] == colour]



            else:
                # colour_key_for_nodes[colour] = [(k, graph.nodes[k]) for k in
                #                                    greedy_colouring_output_dic.keys()
                #                                 if greedy_colouring_output_dic[k] == colour]

                colour_key_for_nodes[colour] = [{k: graph.nodes[k]} for k in
                                                   greedy_colouring_output_dic.keys()
                                                if greedy_colouring_output_dic[k] == colour]


        List_of_Coloured_Graphs_dicts.append(colour_key_for_nodes)

        if plot_graph == True:
            import matplotlib.cm as cm
            plt.figure()
            colour_list = cm.rainbow(np.linspace(0, 1, len(colour_key_for_nodes)))
            pos = nx.circular_layout(graph)

            for colour in colour_key_for_nodes:
                nx.draw_networkx_nodes(graph, pos,
                                       nodelist=[P_word for i in colour_key_for_nodes[colour]
                                                 for P_word, const in i.items()],
                                       node_color=colour_list[colour],
                                       node_size=500,
                                       alpha=0.8
                                     )
                labels = {P_word: P_word for i in colour_key_for_nodes[colour] for P_word, const in i.items()}
                nx.draw_networkx_labels(graph, pos, labels)  # , font_size=8)

            nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
            plt.show()

    return List_of_Coloured_Graphs_dicts

def Get_unique_graph_colours(List_of_Coloured_Graphs_dicts):
    overall_colours={}
    iter = 0
    for colour_dictionary in List_of_Coloured_Graphs_dicts:
        for colour, Nodes in colour_dictionary.items():
            overall_colours[iter] = Nodes
            iter += 1
    return overall_colours

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

    stripped_set={}
    for key in coloured_sets:
        SET = coloured_sets[key]

        stripped = [(PauliWord, const[dict_str_label]) for DICT in SET for PauliWord, const in
                          DICT.items()]

        stripped_set[key] = stripped
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
     [(0, 'I'), (1, 'Z'), (2, 'Z'), (3, 'I')],
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
    G = Build_Graph_Edges_COMMUTING_QWC_AntiCommuting(G, List_of_nodes, 'C', plot_graph = True)

    #comp_G = Get_Complemenary_Graph(G, node_attributes_dict=node_attributes_dict, plot_graph=True) # <- not currently used


    single_G, multi_G = Get_subgraphs(G, node_attributes_dict=node_attributes_dict)
    s_colour = Colour_list_of_Graph(single_G, attribute_dictionary=attribute_dictionary, plot_graph=False,
                                    strategy='largest_first')
    m_colour = Colour_list_of_Graph(multi_G, attribute_dictionary=attribute_dictionary, plot_graph=False,
                                    strategy='largest_first')

    anti_commuting_set = Get_unique_graph_colours(s_colour + m_colour)
    print(anti_commuting_set)


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


    single_G, multi_G = Get_subgraphs(G, node_attributes_dict=node_attributes_dict)
    s_colour = Colour_list_of_Graph(single_G, attribute_dictionary=attribute_dictionary, plot_graph=False,
                                    strategy='largest_first')
    m_colour = Colour_list_of_Graph(multi_G, attribute_dictionary=attribute_dictionary, plot_graph=False,
                                    strategy='largest_first')

    anti_commuting_set = Get_unique_graph_colours(s_colour + m_colour)

    anti_commuting_set_stripped = Get_PauliWord_constant_tuples(anti_commuting_set, dict_str_label='Cofactors')

    print(anti_commuting_set_stripped)


# class Hamiltonian_Graph():
#
#     def __init__(self, List_of_nodes, Graph_colouring_strategy='largest_first', attribute_dictionary=None):
#
#         self.List_of_nodes = List_of_nodes
#         self.Graph_colouring_strategy = Graph_colouring_strategy
#         # self.index_node_and_connected_node_index = index_node_and_connected_node_index
#         self.attribute_dictionary = attribute_dictionary
#
#         self.Graph = nx.Graph()
#         self.anti_commuting_set = None
#         self.node_attributes_dict = None
#
#     def _Get_node_attributes_dict(self):
#         List_of_nodes, node_attributes_dict = Get_list_of_nodes_and_attributes(self.List_of_nodes,
#                                                                         attribute_dictionary=self.attribute_dictionary)
#         self.node_attributes_dict = node_attributes_dict
#
#     def _Build_Graph_nodes(self, plot_graph=False):
#
#         if self.attribute_dictionary is not None:
#             self._Get_node_attributes_dict()
#
#         self.Graph = Build_Graph_Nodes(self.List_of_nodes, self.Graph, node_attributes_dict=self.node_attributes_dict,
#                                        plot_graph=plot_graph)
#
#     def _Build_Graph_edges(self, commutativity_flag, plot_graph=False):
#         self.Graph = Build_Graph_Edges_COMMUTING_QWC_AntiCommuting(self.Graph, self.List_of_nodes, commutativity_flag,
#                                                                    plot_graph=plot_graph)
#
#     def _Colour_Graph(self, plot_graph=False):
#         single_G, multi_G = Get_subgraphs(self.Graph, node_attributes_dict=self.node_attributes_dict)
#         s_colour = Colour_list_of_Graph(single_G, attribute_dictionary=self.node_attributes_dict, plot_graph=plot_graph,
#                                         strategy=self.Graph_colouring_strategy)
#         m_colour = Colour_list_of_Graph(multi_G, attribute_dictionary=self.node_attributes_dict, plot_graph=plot_graph,
#                                         strategy=self.Graph_colouring_strategy)
#
#         output_sets =  Get_unique_graph_colours(s_colour + m_colour)
#         return output_sets
#
#     def Get_Pauli_grouping(self, commutativity_flag, plot_graph=False):
#         self.Graph.clear()
#         self._Build_Graph_nodes(plot_graph=plot_graph)
#         self._Build_Graph_edges(commutativity_flag, plot_graph=plot_graph)
#         output_sets = self._Colour_Graph(plot_graph=plot_graph)
#         return output_sets

# # TODO
#  Find reduction term which has best properties
# 1. maximise commuting terms
# 2. maximise anti-commuting terms
# 3. get QWC terms!

## allows one to measure output all at once



# def Get_commuting_indices(anti_commuting_dict):
#     """
#     Method takes in qubit Hamiltonian as a list of Pauliwords that are lists of tuples (qubitNo, PauliString).
#     Returns each index in qubit Hamiltonian and a list of corresponding indices that the PauliWord commutes with.
#
#     Args:
#         Pauliwords_string_list (list):
#
#     Returns:
#         Commuting_indices (list):
#
#     anti_commuting_dict =
#
#             {
#                  0: [('I0 I1 I2 I3 I4 I5 I6 I7 I8 I9 I10 I11', (-3.9344419569678526+0j))],
#                  1: [('I0 I1 I2 I3 I4 I5 Z6 Z7 I8 I9 I10 I11', (0.07823637778985244+0j))],
#                  2: [('I0 I1 I2 I3 I4 I5 I6 I7 Z8 Z9 I10 I11', (0.07823637778985248+0j))],
#                  3: [('I0 I1 I2 I3 I4 I5 Z6 I7 I8 I9 I10 I11', (-0.2167542932500053+0j)),
#                   ('I0 I1 I2 I3 I4 I5 Y6 X7 X8 Y9 I10 I11', (0.004217284878422762+0j)),
#                   ('Y0 Y1 I2 I3 I4 I5 X6 X7 I8 I9 I10 I11', (-0.0024727061538815324+0j)),
#                   ('Y0 Z1 Z2 Y3 I4 I5 X6 X7 I8 I9 I10 I11', (0.002077887498395716+0j)),
#                   ('Y0 Z1 Z2 Z3 Z4 Y5 X6 X7 I8 I9 I10 I11', (0.002562389780011484+0j)),
#                   ('Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11', (0.0009084689616229723+0j))],
#                  4: [('I0 I1 I2 I3 I4 I5 I6 Z7 I8 I9 I10 I11', (-0.2167542932500053+0j)),
#                   ('I0 I1 I2 I3 I4 I5 Y6 Y7 X8 X9 I10 I11', (-0.004217284878422762+0j)),
#                   ('Y0 X1 I2 I3 I4 I5 X6 Y7 I8 I9 I10 I11', (0.0024727061538815324+0j)),
#                   ('Y0 Z1 Z2 X3 I4 I5 X6 Y7 I8 I9 I10 I11', (-0.002077887498395716+0j)),
#                   ('Y0 Z1 Z2 Z3 Z4 X5 X6 Y7 I8 I9 I10 I11', (-0.002562389780011484+0j)),
#                   ('Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11', (0.0009084689616229723+0j))],
#                  5: [('I0 I1 I2 I3 I4 I5 I6 I7 Z8 I9 I10 I11', (-0.21675429325000534+0j)),
#                   ('I0 I1 I2 I3 I4 I5 X6 X7 Y8 Y9 I10 I11', (-0.004217284878422762+0j)),
#                   ('Y0 X1 I2 I3 I4 I5 I6 I7 X8 Y9 I10 I11', (0.0024727061538815324+0j)),
#                   ('Y0 Z1 Z2 X3 I4 I5 I6 I7 X8 Y9 I10 I11', (-0.002077887498395716+0j)),
#                   ('Y0 Z1 Z2 Z3 Z4 X5 I6 I7 X8 Y9 I10 I11', (-0.002562389780011484+0j))
#               }
#
#
#
#     Returns a List of Tuples that have index of PauliWord and index of terms in the Hamiltonian that it commutes with
#
#     index_of_commuting_terms =
#
#         [(0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
#          (1, [0, 2, 3, 4, 5, 10, 11, 12, 13, 14]),
#          (2, [0, 1, 3, 4, 5, 10, 11, 12, 13, 14]),
#          (3, [0, 1, 2, 4, 5, 10, 11, 12, 13, 14]),
#          (4, [0, 1, 2, 3, 5, 10, 11, 12, 13, 14]),
#          (5, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
#          (6, [0, 5, 7, 8, 9, 10, 11, 12, 13, 14]),]
#
#     """
#     output_dict={}
#     for key in anti_commuting_dict:
#
#         other_keys = [other_key for other_key in anti_commuting_dict if
#                       other_key != key]
#
#         for i in tqdm( range(len(anti_commuting_dict[key])), ascii=True, desc='Getting Commuting Terms'):
#         # for i in range(len(anti_commuting_dict[key])):
#             selected_PauliWord = anti_commuting_dict[key][i]
#
#             Commuting_indices_all = []
#             temp_dict = {}
#
#             for k in other_keys:
#                 checker = np.zeros(len(selected_PauliWord[0]))
#                 for j in range(len(anti_commuting_dict[k])):
#
#                     commuting_indices_for_terms=[]
#
#                     Comparison_PauliWord = anti_commuting_dict[k][j]
#
#                     for x in range(len(selected_PauliWord[0])):
#                         if selected_PauliWord[0][x] == Comparison_PauliWord[0][x]:
#                             checker[x]=1
#                         elif selected_PauliWord[0][x] == 'I' or  Comparison_PauliWord[0][x] == 'I':
#                             checker[x]=1
#                         else:
#                             checker[x]=-1
#
#                         if reduce((lambda x, y: x * y), checker) == 1:  # <----- changing this to -ve one gives anti-commuting
#                             commuting_indices_for_terms.append((k,x))
#
#                         if commuting_indices_for_terms != []:
#                             Commuting_indices_all.append([(key, i), (k, commuting_indices_for_terms)])
#
#                             temp_dict.update({'key': key,
#                                               'index': i,
#
#                                               'related_key_indices': {#'k': k,
#                                                           'related_key_indices': commuting_indices_for_terms}})
#
#     output_dict[key] = temp_dict
#
#
# ###
#
#     index_of_commuting_terms = []
#     for i in range(len(QubitHamiltonianCompleteTerms)):
#         Selected_PauliWord = QubitHamiltonianCompleteTerms[i]
#
#         Complete_index_list = [index for index in range(len(QubitHamiltonianCompleteTerms)) if
#                                index != i]  # all indexes except selected Pauli Word
#
#         Commuting_indexes = []
#         for j in Complete_index_list:
#             j_list = []
#             Comparison_PauliWord = QubitHamiltonianCompleteTerms[j]
#
#             checker = [0 for i in range(len(Selected_PauliWord))]
#             for k in range(len(Selected_PauliWord)):
#                 # compare tuples
#                 if Selected_PauliWord[k] == Comparison_PauliWord[k]:
#                     checker[k] = 1
#
#                 # compare if identity present in selected P word OR of I present in comparison Pauli
#                 elif Selected_PauliWord[k][1] == 'I' or Comparison_PauliWord[k][1] == 'I':
#                     checker[k] = 1
#
#                 else:
#                     checker[k] = -1
#
#             if reduce((lambda x, y: x * y), checker) == 1:  # <----- changing this to -ve one gives anti-commuting
#                 j_list.append(j)
#
#             # if sum(checker) == self.MolecularHamiltonian.n_qubits:
#             #     j_list.append(j)
#
#             if j_list != []:
#                 Commuting_indexes.append(*j_list)
#             else:
#                 # Commuting_indexes.append(j_list)      # <--- commented out! uneeded memory taken
#                 continue
#         commuting_Terms_indices = (i, Commuting_indexes)
#
#         index_of_commuting_terms.append(commuting_Terms_indices)
#
#     return index_of_commuting_terms