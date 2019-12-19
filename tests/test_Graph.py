from quchem.Graph import *
import pytest

# py.test test_Graph.py -vv

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

def test_Get_PauliWords_as_nodes_NORMAL():
    # normal use case
    List_PauliWords = [
        [(0, 'I'), (1, 'I'), (2, 'I'), (3, 'I')],
        [(0, 'Z'), (1, 'I'), (2, 'I'), (3, 'I')],
        [(0, 'I'), (1, 'Z'), (2, 'I'), (3, 'I')],
        [(0, 'I'), (1, 'I'), (2, 'I'), (3, 'Z')],
        [(0, 'I'), (1, 'I'), (2, 'Z'), (3, 'Z')]
    ]
    expected_output = ['I0 I1 I2 I3', 'Z0 I1 I2 I3', 'I0 Z1 I2 I3', 'I0 I1 I2 Z3', 'I0 I1 Z2 Z3']

    output = Get_PauliWords_as_nodes(List_PauliWords)
    assert(output == expected_output)

def test_Get_list_of_nodes_and_attributes_NORMAL():

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
    L_nodes, node_attributes_dict = Get_list_of_nodes_and_attributes(List_of_nodes, attribute_dictionary)

    attrib_dict={}
    for i in range(len(List_of_nodes)):
        temp_dict={}
        for key in attribute_dictionary:
            temp_dict[key] = attribute_dictionary[key][i]
        attrib_dict[List_of_nodes[i]] = temp_dict

    assert (List_of_nodes == L_nodes) and (attrib_dict == node_attributes_dict)


