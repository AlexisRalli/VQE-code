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

        self.G_index = nx.Graph()
        self.node_index_set = None
        self.G_index_comp = None


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



    def Build_nodes(self, string_nodes = True, index_nodes = False, plot_graph = False):

        if string_nodes == True:
            for string_node in self.node_string_set:
                self.G.add_node(string_node)

        elif index_nodes == True:
            for numerical_node in self.node_index_set:
                self.G.add_node(numerical_node)




        if plot_graph == True:
            plt.figure()
            nx.draw(self.G, with_labels=1)
            plt.show()



    def Build_edges(self, plot_graph = False):

        nodes_list = list(self.G.nodes())
        for index, commuting_indices in self.indices:
            for commuting_index in commuting_indices:
                if commuting_index != []:
                    self.G.add_edge(nodes_list[index], nodes_list[commuting_index])

        if plot_graph == True:
            plt.figure()
            pos = nx.circular_layout(self.G)
            nx.draw(self.G, pos, with_labels=1)
            plt.show()


    def Get_complementary_graph(self, plot_graph = False):

        self.G_comp = nx.complement(self.G)

        if plot_graph == True:
            plt.figure()
            pos = nx.circular_layout(self.G_comp)
            nx.draw(self.G_comp, pos, with_labels=1)
            plt.show()


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
    Y.Get_node_terms_as_strings()
    print(Y.node_string_set)
    Y.Build_nodes(plot_graph=True)
    Y.Build_edges(plot_graph=True)
    Y.Get_complementary_graph(plot_graph=True)

