import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class BuildGraph():

    def __init__(self, PauliWords, indices):

        self.PauliWords = PauliWords
        self.indices = indices


        self.G_string = nx.Graph() # undirected graph
        self.node_string_set = None
        self.G_string_comp = None
        self.greedy_string = None
        self.colour_key_for_nodes_string = None

        self.G_index = nx.Graph()
        self.node_index_set = None
        self.G_index_comp = None
        self.greedy_index = None


    def Get_node_terms_as_strings(self):

        node_string_set = []
        for index, commuting_indices in self.indices:
            PauliWord = self.PauliWords[index]
            PauliStrings = ['{}{}'.format(qubitOp, qubitNo) for qubitNo, qubitOp in PauliWord]

            seperator = ' '
            node_string_set.append(seperator.join(PauliStrings))

        self.node_string_set = node_string_set


    def Get_nodes_terms_as_indices(self):
        node_index_set = [index for index, commuting_indices in self.indices]
        self.node_index_set = node_index_set



    def Build_string_nodes(self, plot_graph = False):

        if self.node_string_set == None:
            self.Get_node_terms_as_strings()

        for string_node in self.node_string_set:
            self.G_string.add_node(string_node)


        if plot_graph == True:
            plt.figure()
            nx.draw(self.G_string, with_labels=1)
            plt.show()

    def Build_index_nodes(self, plot_graph = False):

        if self.node_index_set == None:
            self.Get_nodes_terms_as_indices()

        for numerical_node in self.node_index_set:
            self.G_index.add_node(numerical_node)


        if plot_graph == True:
            plt.figure()
            nx.draw(self.G_index, with_labels=1)
            plt.show()



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


    def Build_index_edges(self, plot_graph=False):

        if len(self.G_index.nodes()) == 0:
            self.Build_index_nodes()

        for index, commuting_indices in self.indices:
            for i in range(len(commuting_indices)):

                if commuting_indices[i] != []:
                    self.G_index.add_edge(index, commuting_indices[i])

        if plot_graph == True:
            plt.figure()
            pos = nx.circular_layout(self.G_index)
            nx.draw(self.G_index, pos, with_labels=1)
            plt.show()



    def Get_complementary_graph_string(self, plot_graph = False):

        if len(list(self.G_string.edges())) == 0:
            self.Build_string_edges()

        self.G_string_comp = nx.complement(self.G_string)

        if plot_graph == True:
            plt.figure()
            pos = nx.circular_layout(self.G_string_comp)
            nx.draw(self.G_string_comp, pos, with_labels=1)
            plt.show()

    def Get_complementary_graph_index(self, plot_graph = False):

        if len(list(self.G_index.edges())) == 0:
            self.Build_index_edges()

        self.G_index_comp = nx.complement(self.G_index)

        if plot_graph == True:
            plt.figure()
            pos = nx.circular_layout(self.G_index_comp)
            nx.draw(self.G_index_comp, pos, with_labels=1)
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
            colour_key_for_nodes_string[colour] = [k for k in self.greedy_string.keys()
                                                   if self.greedy_string[k] == colour]
        self.colour_key_for_nodes_string = colour_key_for_nodes_string

        if plot_graph == True:
            import matplotlib.cm as cm
            plt.figure()
            colour_list = cm.rainbow(np.linspace(0, 1, len(self.colour_key_for_nodes_string)))
            pos = nx.circular_layout(self.G_string_comp)

            for colour in self.colour_key_for_nodes_string:
                nx.draw_networkx_nodes(self.G_string_comp, pos,
                                       nodelist=self.colour_key_for_nodes_string[colour],
                                       node_color=colour_list[colour],
                                       node_size=500,
                                       alpha=0.8)

            nx.draw_networkx_edges(self.G_string_comp, pos, width=1.0, alpha=0.5)

            # need to get FONT to change! TODO
            nx.draw_networkx_labels(self.G_string_comp, pos, font_family='Times-New-Roman', font_size=12)
            plt.plot()

    def Get_coloured_keys_index(self, plot_graph=False):
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
            pos = nx.circular_layout(self.G_index_comp)

            for colour in self.colour_key_for_nodes_index:
                nx.draw_networkx_nodes(self.G_index_comp, pos,
                                       nodelist=self.colour_key_for_nodes_index[colour],
                                       node_color=colour_list[colour],
                                       node_size=500,
                                       alpha=0.8)

            nx.draw_networkx_edges(self.G_index_comp, pos, width=1.0, alpha=0.5)

            nx.draw_networkx_labels(self.G_index_comp, pos)
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
     (5, [0, 1, 2, 3, 4, [], [], [], [], 10, 11, 12, 13, 14]),
     (6, [0, [], [], [], [], [], [], [], [], [], [], [], [], []]),
     (7, [0, [], [], [], [], [], [], [], [], [], [], [], [], []]),
     (8, [0, [], [], [], [], [], [], [], [], [], [], [], [], []]),
     (9, [0, [], [], [], [], [], [], [], [], [], [], [], [], []]),
     (10, [0, 1, 2, 3, 4, 5, [], [], [], [], 11, 12, 13, 14]),
     (11, [0, 1, 2, 3, 4, 5, [], [], [], [], 10, 12, 13, 14]),
     (12, [0, 1, 2, 3, 4, 5, [], [], [], [], 10, 11, 13, 14]),
     (13, [0, 1, 2, 3, 4, 5, [], [], [], [], 10, 11, 12, 14]),
     (14, [0, 1, 2, 3, 4, 5, [], [], [], [], 10, 11, 12, 13])]

    Y = BuildGraph(PauliWords, indices)
    Y.Build_index_nodes()# plot_graph=True)
    Y.Build_index_edges()# plot_graph=True)
    Y.Get_complementary_graph_index() #plot_graph=True)
    Y.colour_index_graph()
    Y.Get_coloured_keys_index(plot_graph=True)

    Y.Build_string_nodes()# plot_graph=True)
    Y.Build_string_edges()# plot_graph=True)
    Y.Get_complementary_graph_string()# plot_graph=True)
    Y.colour_string_graph()
    Y.Get_coloured_keys_string(plot_graph=True)

