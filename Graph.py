import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class BuildGraph():

    def __init__(self, PauliWords, indices, HamiltonainCofactors):

        self.PauliWords = PauliWords
        self.indices = indices
        self.HamiltonainCofactors = HamiltonainCofactors

        self.G = nx.Graph()
        self.node_index_set = None
        self.index_string_dict = None
        self.G_comp = None
        self.greedy_index = None



    def Get_nodes_terms_as_indices(self):

        """
        Function takes indexes and

        For example:

        self.indices =
        [
            (0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
            (1, [0, 2, 3, 4, 5, [], [], [], [], 10, 11, 12, 13, 14]),
            (2, [0, 1, 3, 4, 5, [], [], [], [], 10, 11, 12, 13, 14]),
            (3, [0, 1, 2, 4, 5, [], [], [], [], 10, 11, 12, 13, 14]),
            (4, [0, 1, 2, 3, 5, [], [], [], [], 10, 11, 12, 13, 14]),
            (5, [0, 1, 2, 3, 4, [], [], [], [], 10, 11, 12, 13, 14]),
            (6, [0, [], [], [], [], [], [], [], [], [], [], [], [], []]),
            (7, [0, [], [], [], [], [], [], [], [], [], [], [], [], []]),
            (8, [0, [], [], [], [], [], [], [], [], [], [], [], [], []]),
            (9, [0, [], [], [], [], [], [], [], [], [], [], [], [], []]),
            (10, [0, 1, 2, 3, 4, 5, [], [], [], [], 11, 12, 13, 14]),
            (11, [0, 1, 2, 3, 4, 5, [], [], [], [], 10, 12, 13, 14]),
            (12, [0, 1, 2, 3, 4, 5, [], [], [], [], 10, 11, 13, 14]),
            (13, [0, 1, 2, 3, 4, 5, [], [], [], [], 10, 11, 12, 14]),
            (14, [0, 1, 2, 3, 4, 5, [], [], [], [], 10, 11, 12, 13])
        ]


        self.node_index_set =
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

        """

        node_index_set = [index for index, commuting_indices in self.indices]
        self.node_index_set = node_index_set


    def Get_string_indexes(self):

        """
        Function goes through PauliWords and appends to node key (self.G.nodes) a string version of the PauliWord
        Output is a dictionary... where key corresponds to node label (self.G.nodes)...
        The graph is usually plotted with these index keys (integars) this dictionary allows the graph to be
        plotted with the PauliWords instead!

        self.PauliWords=
        [
             [(0, 'I'), (1, 'I'), (2, 'I'), (3, 'I')],
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
             [(0, 'I'), (1, 'I'), (2, 'Z'), (3, 'Z')]
         ]

        self.index_string_dict =
        {
             0: 'I0 I1 I2 I3',
             1: 'Z0 I1 I2 I3',
             2: 'I0 Z1 I2 I3',
             3: 'I0 I1 Z2 I3',
             4: 'I0 I1 I2 Z3',
             5: 'Z0 Z1 I2 I3',
             6: 'Y0 X1 X2 Y3',
             7: 'Y0 Y1 X2 X3',
             8: 'X0 X1 Y2 Y3',
             9: 'X0 Y1 Y2 X3',
             10: 'Z0 I1 Z2 I3',
             11: 'Z0 I1 I2 Z3',
             12: 'I0 Z1 Z2 I3',
             13: 'I0 Z1 I2 Z3',
             14: 'I0 I1 Z2 Z3'
         }

        :return:
        """

        if self.node_index_set == None:
            self.Get_nodes_terms_as_indices()

        index_string_dict = {}
        for key in list(self.G.nodes):
            PauliWord = self.PauliWords[key]
            PauliStrings = ['{}{}'.format(qubitOp, qubitNo) for qubitNo, qubitOp in PauliWord]

            seperator = ' '
            together = seperator.join(PauliStrings)
            index_string_dict.update({key: together})

        self.index_string_dict = index_string_dict


    def Build_index_nodes(self, plot_graph = False):

        if self.node_index_set == None:
            self.Get_nodes_terms_as_indices()

        for numerical_node in self.node_index_set:
            self.G.add_node(numerical_node)


        if plot_graph == True:
            plt.figure()
            nx.draw(self.G, with_labels=1)
            plt.show()




    def Build_index_edges(self, plot_graph=False):
        """

        self.indices =
        [
            (0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
            (1, [0, 2, 3, 4, 5, [], [], [], [], 10, 11, 12, 13, 14]),
            (2, [0, 1, 3, 4, 5, [], [], [], [], 10, 11, 12, 13, 14]),
            (3, [0, 1, 2, 4, 5, [], [], [], [], 10, 11, 12, 13, 14]),
            (4, [0, 1, 2, 3, 5, [], [], [], [], 10, 11, 12, 13, 14]),
            (5, [0, 1, 2, 3, 4, [], [], [], [], 10, 11, 12, 13, 14]),
            (6, [0, [], [], [], [], [], [], [], [], [], [], [], [], []]),
            (7, [0, [], [], [], [], [], [], [], [], [], [], [], [], []]),
            (8, [0, [], [], [], [], [], [], [], [], [], [], [], [], []]),
            (9, [0, [], [], [], [], [], [], [], [], [], [], [], [], []]),
            (10, [0, 1, 2, 3, 4, 5, [], [], [], [], 11, 12, 13, 14]),
            (11, [0, 1, 2, 3, 4, 5, [], [], [], [], 10, 12, 13, 14]),
            (12, [0, 1, 2, 3, 4, 5, [], [], [], [], 10, 11, 13, 14]),
            (13, [0, 1, 2, 3, 4, 5, [], [], [], [], 10, 11, 12, 14]),
            (14, [0, 1, 2, 3, 4, 5, [], [], [], [], 10, 11, 12, 13])
        ]

        therefore if commuting_indices != [] then it will commute with that term (indexed)!

        :param plot_graph:
        :return:
        """

        if len(self.G.nodes()) == 0:
            self.Build_index_nodes()

        for index, commuting_indices in self.indices:
            for i in range(len(commuting_indices)):

                if commuting_indices[i] != []:
                    self.G.add_edge(index, commuting_indices[i])

        if plot_graph == True:
            plt.figure()
            pos = nx.circular_layout(self.G)
            nx.draw(self.G, pos, with_labels=1)
            plt.show()



    def Get_complementary_graph_index(self, plot_graph = False):

        if len(list(self.G.edges())) == 0:
            self.Build_index_edges()

        self.G_comp = nx.complement(self.G)

        if plot_graph == True:
            plt.figure()
            pos = nx.circular_layout(self.G_comp)
            nx.draw(self.G_comp, pos, with_labels=1)
            plt.show()


    def plot_graph_with_strings(self, graph):

        if self.index_string_dict == None:
            self.Get_string_indexes()

        plt.figure()
        pos = nx.circular_layout(graph)
        nx.draw_networkx_nodes(graph, pos)
        nx.draw_networkx_labels(graph, pos, labels=self.index_string_dict)
        plt.show()

    def colour_index_graph(self, strategy='largest_first'):

        if self.G_comp == None:
            self.Get_complementary_graph_index()

        self.greedy_index = nx.greedy_color(self.G_comp, strategy=strategy, interchange=False)



    def Get_coloured_keys_index(self, plot_graph=False, string_graph = False):
        # store the names (the keys of the new dict) as a set (keeps elements unique)
        unique_colours = set(self.greedy_index.values())

        # use a list comprehension, iterating through keys and checking the values match each k
        colour_key_for_nodes_index = {}
        for colour in unique_colours:
            colour_key_for_nodes_index[colour] = [k for k in self.greedy_index.keys()
                                                  if self.greedy_index[k] == colour]
        self.colour_key_for_nodes_index = colour_key_for_nodes_index

        if plot_graph == True:
            import matplotlib.cm as cm

            plt.figure()
            colour_list = cm.rainbow(np.linspace(0, 1, len(self.colour_key_for_nodes_index)))
            pos = nx.circular_layout(self.G_comp)

            for colour in self.colour_key_for_nodes_index:
                nx.draw_networkx_nodes(self.G_comp, pos,
                                       nodelist=self.colour_key_for_nodes_index[colour],
                                       node_color=colour_list[colour],
                                       node_size=500,
                                       alpha=0.8)

            nx.draw_networkx_edges(self.G_comp, pos, width=1.0, alpha=0.5)

            if string_graph == True:

                if self.index_string_dict == None:
                    self.Get_string_indexes()
                nx.draw_networkx_labels(self.G_comp, pos, labels=self.index_string_dict)
            else:
                nx.draw_networkx_labels(self.G_comp, pos)
            plt.plot()


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
               (1, [0, 2, 3, 4, 5, [], [], [], [], 10, 11, 12, 13, 14]),
               (2, [0, 1, 3, 4, 5, [], [], [], [], 10, 11, 12, 13, 14]),
               (3, [0, 1, 2, 4, 5, [], [], [], [], 10, 11, 12, 13, 14]),
               (4, [0, 1, 2, 3, 5, [], [], [], [], 10, 11, 12, 13, 14]),
               (5, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
               (6, [0, [], [], [], [], 5, 7, 8, 9, 10, 11, 12, 13, 14]),
               (7, [0, [], [], [], [], 5, 6, 8, 9, 10, 11, 12, 13, 14]),
               (8, [0, [], [], [], [], 5, 6, 7, 9, 10, 11, 12, 13, 14]),
               (9, [0, [], [], [], [], 5, 6, 7, 8, 10, 11, 12, 13, 14]),
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

    Y = BuildGraph(PauliWords, indices, HamiltonainCofactors)
    Y.Build_index_nodes()# plot_graph=True)
    Y.Build_index_edges()# plot_graph=True)
    Y.Get_complementary_graph_index() #plot_graph=True)
    Y.colour_index_graph()
    Y.Get_coloured_keys_index(plot_graph=True)

    #Y.plot_graph_with_strings(Y.G)
    Y.Get_coloured_keys_index(plot_graph=True, string_graph=True)
    print(Y.colour_key_for_nodes_index)


#########

class BuildGraph_string():

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

        if self.node_string_set == None:
            self.Get_node_terms_as_strings()

        for string_node in self.node_string_set:
            self.G_string.add_node(string_node)


        if plot_graph == True:
            plt.figure()
            nx.draw(self.G_string, with_labels=1)
            plt.show()


        if plot_graph == True:
            plt.figure()
            nx.draw(self.G_index, with_labels=1)
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


    def colour_string_graph(self, strategy = 'largest_first'):

        if self.G_string_comp == None:
            self.Get_complementary_graph_string()

        self.greedy_string = nx.greedy_color(self.G_string_comp, strategy=strategy, interchange=False)

    def colour_index_graph(self, strategy='largest_first'):

        if self.G_index_comp == None:
            self.Get_complementary_graph_index()

        self.greedy_index = nx.greedy_color(self.G_index_comp, strategy=strategy, interchange=False)


    def Get_coloured_keys_string(self, plot_graph = False):


        # store the names (the keys of the new dict) as a set (keeps elements unique)
        unique_colours = set(self.greedy_string.values())

        colour_key_for_nodes_string = {}
        for colour in unique_colours:
            colour_key_for_nodes_string[colour] = [(k, self.G_string.nodes[k]['Cofactor']) for k in self.greedy_string.keys()
                                                   if self.greedy_string[k] == colour]
        self.colour_key_for_nodes_string = colour_key_for_nodes_string


        if plot_graph == True:
            import matplotlib.cm as cm
            plt.figure()
            colour_list = cm.rainbow(np.linspace(0, 1, len(self.colour_key_for_nodes_string)))
            pos = nx.circular_layout(self.G_string_comp)

            for colour in self.colour_key_for_nodes_string:
                nx.draw_networkx_nodes(self.G_string_comp, pos,
                                       nodelist=[PauliWord for PauliWord, constant in self.colour_key_for_nodes_string[colour]],
                                       node_color=colour_list[colour],
                                       node_size=500,
                                       alpha=0.8)

            nx.draw_networkx_edges(self.G_string_comp, pos, width=1.0, alpha=0.5)

            # need to get FONT to change! TODO
            nx.draw_networkx_labels(self.G_string_comp, pos, font_family='Times-New-Roman', font_size=12)
            plt.legend()
            plt.plot()

    def GG(self):

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

            >>> G = nx.path_graph(4)
            >>> nx.add_path(G, [10, 11, 12])
            >>> [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
            [4, 3]

            If you only want the largest connected component, it's more
            efficient to use max instead of sort.

            >>> largest_cc = max(nx.connected_components(G), key=len)

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
            >>> G = nx.path_graph(4)
            >>> G.add_edge(5,6)
            >>> graphs = list(nx.connected_component_subgraphs(G))

            If you only want the largest connected component, it's more
            efficient to use max instead of sort:

            >>> Gc = max(nx.connected_component_subgraphs(G), key=len)

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
            >>> G = nx.path_graph(4)
            >>> print(nx.is_connected(G))
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

        connected_graphs = list(connected_component_subgraphs(self.G_string_comp))
        multi_node_G = []
        single_node_G = []
        for graph in connected_graphs:

            if len(graph.nodes) > 1:
                # pos = nx.circular_layout(graph)
                # plt.figure()
                # nx.draw(graph, pos, with_labels=1)
                multi_node_G.append(graph)
            else:
                single_node_G.append(graph)
        return single_node_G + multi_node_G

    def colouring(self):

        graphs = self.GG()

        for comp_graph in graphs:

            strategy = 'independent_set'
            greedy_string = nx.greedy_color(comp_graph, strategy=strategy, interchange=False)
            #greedy_index = nx.greedy_color(comp_graph, strategy=strategy, interchange=False)

            unique_colours = set(greedy_string.values())

            colour_key_for_nodes_string = {}
            for colour in unique_colours:
                colour_key_for_nodes_string[colour] = [k for k in greedy_string.keys()
                                                       if greedy_string[k] == colour]
            colour_key_for_nodes_string = colour_key_for_nodes_string

            import matplotlib.cm as cm

            plt.figure()
            colour_list = cm.rainbow(np.linspace(0, 1, len(colour_key_for_nodes_string)))
            pos = nx.circular_layout(comp_graph)

            for colour in colour_key_for_nodes_string:
                nx.draw_networkx_nodes(comp_graph, pos,
                                       nodelist=[PauliWord for PauliWord in colour_key_for_nodes_string[colour]],
                                       node_color=colour_list[colour],
                                       node_size=500,
                                       alpha=0.8)

            nx.draw_networkx_edges(comp_graph, pos, width=1.0, alpha=0.5)

            # need to get FONT to change! TODO
            nx.draw_networkx_labels(comp_graph, pos, font_family='Times-New-Roman', font_size=12)
            plt.plot()

            nx.set_node_attributes(self.G_string, self.node_string_set_and_HamiltonainCofactors, 'Cofactor')



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

    X.Build_string_nodes()# plot_graph=True)
    X.Build_string_edges()# plot_graph=True)
    X.Get_complementary_graph_string()# plot_graph=True)
    X.colour_string_graph()
    X.Get_coloured_keys_string(plot_graph=True)
    print(X.colour_key_for_nodes_string)



# from Hamiltonian_Generator_Functions import Hamiltonian
# if __name__ == '__main__':
#
#     Z = Hamiltonian('BeH2')
#     Z.Get_all_info(get_FCI_energy=False)
#     PauliWords = Z.QubitHamiltonianCompleteTerms
#     indices = Z.Commuting_indices
#     HamiltonainCofactors = Z.HamiltonainCofactors
#
#
#     LO = BuildGraph_string(PauliWords, indices, HamiltonainCofactors)
#
#     LO.Build_string_nodes()# plot_graph=True)
#     LO.Build_string_edges()# plot_graph=True)
#     LO.Get_complementary_graph_string()# plot_graph=True)
#     LO.max_clique_cover_composite_graph()
#     LO.max_clique_cover











# G = X.G_string
# comp_G = X.G_string_comp
#
# #https://stackoverflow.com/questions/29127350/completely-connected-subgraphs-from-a-larger-graph-in-networkx
# list(nx.clique.find_cliques(comp_G))
#
#
# ll = nx.make_max_clique_graph(comp_G)
# pos = nx.circular_layout(ll)
# plt.figure()
# nx.draw(ll, pos, with_labels=1)
#
#
# from networkx.algorithms.components.connected import connected_component_subgraphs
# graphs = list(connected_component_subgraphs(comp_G))
# #this allows access to connected_component_subgraphs function!!!!
# #https://testfixsphinx.readthedocs.io/en/latest/_modules/networkx/algorithms/components/connected.html#connected_component_subgraphs
#
# for graph in graphs:
#     pos = nx.circular_layout(graph)
#     plt.figure()
#     nx.draw(graph, pos, with_labels=1)
#
# # nx.set_node_attributes(X.G_string, X.node_string_set_and_HamiltonainCofactors, 'Cofactor')
# # X.G_string.nodes['I0 I1 I2 I3']['Cofactor']
#
H = nx.Graph()
H.add_nodes_from([i+1 for i in range(8)])
edg = [
    (1,2), (2,3), (3,1), (1,4), (4,5), (5, 6), (6,4), (5,7), (7,8), (8,3)
]

for e in edg:
    H.add_edge(*e)



########################################################################
H_comp = nx.complement(H)
strategy = 'independent_set'
greedy_string = nx.greedy_color(H_comp, strategy=strategy, interchange=False)
greedy_index = nx.greedy_color(H_comp, strategy=strategy, interchange=False)

unique_colours = set(greedy_string.values())

colour_key_for_nodes_string = {}
for colour in unique_colours:
    colour_key_for_nodes_string[colour] = [k for k in greedy_string.keys()
                                           if greedy_string[k] == colour]
colour_key_for_nodes_string = colour_key_for_nodes_string


import matplotlib.cm as cm

plt.figure()
colour_list = cm.rainbow(np.linspace(0, 1, len(colour_key_for_nodes_string)))
pos = nx.circular_layout(H_comp)

for colour in colour_key_for_nodes_string:
    nx.draw_networkx_nodes(H_comp, pos,
                           nodelist=[PauliWord for PauliWord in colour_key_for_nodes_string[colour]],
                           node_color=colour_list[colour],
                           node_size=500,
                           alpha=0.8)

nx.draw_networkx_edges(H_comp, pos, width=1.0, alpha=0.5)

# need to get FONT to change! TODO
nx.draw_networkx_labels(H_comp, pos, font_family='Times-New-Roman', font_size=12)
plt.plot()
################


graphs = list(connected_component_subgraphs(X.G_string_comp))

multiple_connections_G = []
for graph in graphs:

    if len(graph.nodes) > 1:
        # pos = nx.circular_layout(graph)
        # plt.figure()
        # nx.draw(graph, pos, with_labels=1)
        multiple_connections_G.append(graph)

H_comp = multiple_connections_G[0]
strategy = 'independent_set'
greedy_string = nx.greedy_color(H_comp, strategy=strategy, interchange=False)
greedy_index = nx.greedy_color(H_comp, strategy=strategy, interchange=False)

unique_colours = set(greedy_string.values())

colour_key_for_nodes_string = {}
for colour in unique_colours:
    colour_key_for_nodes_string[colour] = [k for k in greedy_string.keys()
                                           if greedy_string[k] == colour]
colour_key_for_nodes_string = colour_key_for_nodes_string

import matplotlib.cm as cm

plt.figure()
colour_list = cm.rainbow(np.linspace(0, 1, len(colour_key_for_nodes_string)))
pos = nx.circular_layout(H_comp)

for colour in colour_key_for_nodes_string:
    nx.draw_networkx_nodes(H_comp, pos,
                           nodelist=[PauliWord for PauliWord in colour_key_for_nodes_string[colour]],
                           node_color=colour_list[colour],
                           node_size=500,
                           alpha=0.8)

nx.draw_networkx_edges(H_comp, pos, width=1.0, alpha=0.5)

# need to get FONT to change! TODO
nx.draw_networkx_labels(H_comp, pos, font_family='Times-New-Roman', font_size=12)
plt.plot()





connected_graphs = list(connected_component_subgraphs(X.G_string_comp))
multi_node_G = []
single_node_G = []
for graph in connected_graphs:

    if len(graph.nodes) > 1:
        # pos = nx.circular_layout(graph)
        # plt.figure()
        # nx.draw(graph, pos, with_labels=1)
        multi_node_G.append(graph)
    else:
        single_node_G.append(graph)




H_comp = multi_node_G[0]
strategy = 'independent_set'
greedy_string = nx.greedy_color(H_comp, strategy=strategy, interchange=False)
greedy_index = nx.greedy_color(H_comp, strategy=strategy, interchange=False)

unique_colours = set(greedy_string.values())

colour_key_for_nodes_string = {}
for colour in unique_colours:
    colour_key_for_nodes_string[colour] = [k for k in greedy_string.keys()
                                           if greedy_string[k] == colour]
colour_key_for_nodes_string = colour_key_for_nodes_string

import matplotlib.cm as cm

plt.figure()
colour_list = cm.rainbow(np.linspace(0, 1, len(colour_key_for_nodes_string)))
pos = nx.circular_layout(H_comp)

for colour in colour_key_for_nodes_string:
    nx.draw_networkx_nodes(H_comp, pos,
                           nodelist=[PauliWord for PauliWord in colour_key_for_nodes_string[colour]],
                           node_color=colour_list[colour],
                           node_size=500,
                           alpha=0.8)

nx.draw_networkx_edges(H_comp, pos, width=1.0, alpha=0.5)

# need to get FONT to change! TODO
nx.draw_networkx_labels(H_comp, pos, font_family='Times-New-Roman', font_size=12)
plt.plot()












#
#
# nodes = list(H.nodes)
# cliques = list(nx.find_cliques(H))
# sorted_cliques = sorted(cliques, key=len, reverse=True)
# clique_list=[]
# for clique in sorted_cliques:
#     #print(clique)
#     if clique_list ==[]:
#         clique_list.append(clique)
#     else:
#         checker = [i for i in clique for cc in clique_list if i in cc]
#         print(checker)
#
#         if len(checker) > 0:
#             continue
#         else:
#             clique_list.append(clique)
#
#
# length = sum(1 for el in nx.find_cliques(H))
# my_array = np.empty(length, dtype = object)
# for i, el in enumerate(nx.find_cliques(H)):
#     my_array[i] = np.array(el, dtype = object)
# my_array = np.array(sorted(my_array, key=len, reverse=True))


#nx.greedy_color(H,nx.coloring.strategy_independent_set(H))
#
# # nodes = list(H.nodes)
# # cliques = list(nx.find_cliques(H))
# # sorted_cliques = sorted(cliques, key=len, reverse=True)
# # sorted_cliques = [set(i) for i in sorted_cliques]
# #
# # for node in nodes:
# #     for clique in sorted_cliques:
# #
# # clique_list = []
# #
# #
# # for i in range(len(sorted_cliques)):
# #     clique = sorted_cliques[i]
# #     Not_indexed_sets = [node for node in sorted_cliques[i+1::] if clique not in sorted_cliques[i+1::]]
# #
# #     for
# #
# #
# #
# #
# #
# #
# # from functools import reduce
# #
# # nodes = list(H.nodes)
# # cliques = list(nx.find_cliques(H))
# # sorted_cliques = sorted(cliques, key=len, reverse=True)
# # sorted_cliques = [set(i) for i in sorted_cliques]
# # Not_indexed_sets =[1,2]
# # clique_list = []
# # while len(Not_indexed_sets) > 0:
# #     clique = sorted_cliques.pop(0)
# #
# #     clique_list.append(clique)
# #
# #     #print('hello', sorted_cliques)
# #     Not_indexed_sets = [node for node in sorted_cliques[1::] if clique not in sorted_cliques[1::]]
# #     print(Not_indexed_sets)
# #
# #     sorted_cliques = reduce((lambda x, y: x - y), sorted_cliques, Not_indexed_sets)
# #
# #
# # cliques = np.array([np.array(clique) for clique in list(nx.find_cliques(H))])
# #
# # clique_list=[]
# # for clique in cliques:
# #     if clique_list ==[]:
# #         clique_list.append(clique)
# #     else:
# #         for clique_good in clique_list:
# #             if np.isin(clique, clique_good).any():
# #                 clique_list.append(clique)
