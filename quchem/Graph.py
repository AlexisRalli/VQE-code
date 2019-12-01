import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
#from tqdm import tqdm

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

        if self.node_string_set == None:
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

