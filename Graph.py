import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class BuildGraph():

    def __init__(self, PauliWords, indices):

        self.PauliWords = PauliWords
        self.indices = indices

        self.G = nx.Graph()
        self.node_index_set = None
        self.index_string_dict = None
        self.G_comp = None
        self.greedy_index = None



    def Get_nodes_terms_as_indices(self):
        node_index_set = [index for index, commuting_indices in self.indices]
        self.node_index_set = node_index_set


    def Get_string_indexes(self):

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

    #Y.plot_graph_with_strings(Y.G)
    Y.Get_coloured_keys_index(plot_graph=True, string_graph=True)
    print(Y.colour_key_for_nodes_index)

