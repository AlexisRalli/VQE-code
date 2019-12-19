import numpy as np
from functools import reduce
from quchem.Graph import *
#from itertools import zip_longest
List_PauliWords = [[(0, 'I'), (1, 'I'),(2, 'I'),(3, 'I'),(4, 'I'),(5, 'I'),(6, 'I'),(7, 'I'),(8, 'I'),(9, 'I'),(10, 'I'),(11, 'I')],
                   [(0, 'Z'),(1, 'I'),(2, 'I'),(3, 'I'),(4, 'I'),(5, 'I'),(6, 'I'),(7, 'I'),(8, 'I'),(9, 'I'),(10, 'I'),(11, 'I')],
                   [(0, 'Y'),(1, 'Z'),(2, 'Y'),(3, 'I'),(4, 'I'),(5, 'I'),(6, 'I'),(7, 'I'),(8, 'I'),(9, 'I'),(10, 'I'),(11, 'I')],
                   [(0, 'X'),(1, 'Z'),(2, 'X'),(3, 'I'),(4, 'I'),(5, 'I'),(6, 'I'),(7, 'I'),(8, 'I'),(9, 'I'),(10, 'I'),(11, 'I')],
                   [(0, 'Y'),(1, 'Z'),(2, 'Z'),(3, 'Z'),(4, 'Y'),(5, 'I'),(6, 'I'),(7, 'I'),(8, 'I'),(9, 'I'),(10, 'I'),(11, 'I')],
                   [(0, 'X'),(1, 'Z'),(2, 'Z'),(3, 'Z'),(4, 'X'),(5, 'I'),(6, 'I'),(7, 'I'),(8, 'I'),(9, 'I'),(10, 'I'),(11, 'I')],
                   [(0, 'Y'),(1, 'Z'),(2, 'Z'),(3, 'Z'),(4, 'Z'),(5, 'Z'),(6, 'Z'),(7, 'Z'),(8, 'Z'),(9, 'Z'),(10, 'Y'),(11, 'I')],
                   [(0, 'X'),(1, 'Z'),(2, 'Z'),(3, 'Z'),(4, 'Z'),(5, 'Z'),(6, 'Z'),(7, 'Z'),(8, 'Z'),(9, 'Z'),(10, 'X'),(11, 'I')],
                   [(0, 'I'),(1, 'Z'),(2, 'I'),(3, 'I'),(4, 'I'),(5, 'I'),(6, 'I'),(7, 'I'),(8, 'I'),(9, 'I'),(10, 'I'),(11, 'I')],
                   [(0, 'I'),(1, 'Y'),(2, 'Z'),(3, 'Y'),(4, 'I'),(5, 'I'),(6, 'I'),(7, 'I'),(8, 'I'),(9, 'I'),(10, 'I'),(11, 'I')],
                   [(0, 'I'),(1, 'X'),(2, 'Z'),(3, 'X'),(4, 'I'),(5, 'I'),(6, 'I'),(7, 'I'),(8, 'I'),(9, 'I'),(10, 'I'),(11, 'I')],
                   [(0, 'I'),(1, 'Y'),(2, 'Z'),(3, 'Z'),(4, 'Z'),(5, 'Y'),(6, 'I'),(7, 'I'),(8, 'I'),(9, 'I'),(10, 'I'),(11, 'I')],
                   [(0, 'I'),(1, 'X'),(2, 'Z'),(3, 'Z'),(4, 'Z'),(5, 'X'),(6, 'I'),(7, 'I'),(8, 'I'),(9, 'I'),(10, 'I'),(11, 'I')],
                   [(0, 'I'),(1, 'Y'),(2, 'Z'),(3, 'Z'),(4, 'Z'),(5, 'Z'),(6, 'Z'),(7, 'Z'),(8, 'Z'),(9, 'Z'),(10, 'Z'),(11, 'Y')],
                   [(0, 'I'),(1, 'X'),(2, 'Z'),(3, 'Z'),(4, 'Z'),(5, 'Z'),(6, 'Z'),(7, 'Z'),(8, 'Z'),(9, 'Z'),(10, 'Z'),(11, 'X')],
                   [(0, 'I'),(1, 'I'),(2, 'Z'),(3, 'I'),(4, 'I'),(5, 'I'),(6, 'I'),(7, 'I'),(8, 'I'),(9, 'I'),(10, 'I'),(11, 'I')],
                   [(0, 'I'),(1, 'I'),(2, 'Y'),(3, 'Z'),(4, 'Y'),(5, 'I'),(6, 'I'),(7, 'I'),(8, 'I'),(9, 'I'),(10, 'I'),(11, 'I')],
                   [(0, 'I'),(1, 'I'),(2, 'X'),(3, 'Z'),(4, 'X'),(5, 'I'),(6, 'I'),(7, 'I'),(8, 'I'),(9, 'I'),(10, 'I'),(11, 'I')],
                   [(0, 'I'),(1, 'I'),(2, 'Y'),(3, 'Z'),(4, 'Z'),(5, 'Z'),(6, 'Z'),(7, 'Z'),(8, 'Z'),(9, 'Z'),(10, 'Y'),(11, 'I')],
                   [(0, 'I'),(1, 'I'),(2, 'X'),(3, 'Z'),(4, 'Z'),(5, 'Z'),(6, 'Z'),(7, 'Z'),(8, 'Z'),(9, 'Z'),(10, 'X'),(11, 'I')]
                   ]
HamiltonainCofactors = [(-3.9344419569678446+0j),
 (1.04962640047693+0j),
 (-0.023844584591133436+0j),
 (-0.023844584591133436+0j),
 (-0.026332990895885356+0j),
 (-0.026332990895885356+0j),
 (-0.017297109487008907+0j),
 (-0.017297109487008907+0j),
 (1.0496264004769302+0j),
 (-0.023844584591133443+0j),
 (-0.023844584591133443+0j),
 (-0.026332990895885387+0j),
 (-0.026332990895885387+0j),
 (-0.01729710948700891+0j),
 (-0.01729710948700891+0j),
 (-0.09129805365197576+0j),
 (-0.007987782352070982+0j),
 (-0.007987782352070982+0j),
 (-0.005200666861919969+0j),
 (-0.005200666861919969+0j)]


List_of_nodes = Get_PauliWords_as_nodes(List_PauliWords)
attribute_dictionary = {'Cofactors': HamiltonainCofactors}

List_of_nodes, node_attributes_dict = Get_list_of_nodes_and_attributes(List_of_nodes,
                                                                       attribute_dictionary=attribute_dictionary)

G = nx.Graph()
G = Build_Graph_Nodes(List_of_nodes, G, node_attributes_dict=node_attributes_dict, plot_graph=False)
G = Build_Graph_Edges_COMMUTING(G, List_of_nodes, plot_graph=True)

# comp_G = Get_Complemenary_Graph(G, node_attributes_dict=node_attributes_dict, plot_graph=True) # <- not currently used


single_G, multi_G = Get_subgraphs(G, node_attributes_dict=node_attributes_dict)
s_colour = Colour_list_of_Graph(single_G, attribute_dictionary=attribute_dictionary, plot_graph=False,
                                strategy='largest_first')
m_colour = Colour_list_of_Graph(multi_G, attribute_dictionary=attribute_dictionary, plot_graph=False,
                                strategy='largest_first')

anti_commuting_sets = Get_unique_graph_colours(s_colour + m_colour)
print(anti_commuting_sets)



def Commutativity(P1, P2, anti_comm=False):
    """
     Find if two PauliWords either commute or anti_commute.
     By default it will check if they commute.

    Args:
        P1 (str): First PauliWord to compare
        P2 (str): Second PauliWord to compare

    Returns:
        (bool): True or false as to whether terms commute or anti_commute

    .. code-block:: python
       :emphasize-lines: 5

       from TODO import *

       P1 = 'I0 I1 I2 I3 I4 I5 Z6 I7 I8 I9 I10 I11'
       P2 = 'I0 I1 Y2 I3 I4 I5 Z6 I7 I8 I9 X10 I11'
       Commutativity(P1, P2, anti_comm=False)
       >> True
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

    if reduce((lambda x, y: x * y), checker) == 1:
        if anti_comm is False:
            return True
        else:
            return False
    else:
        if anti_comm is False:
            return False
        else:
            return True

# # print(Commutativity(str(*anti_commuting_sets[2][0].keys()), str(*anti_commuting_sets[2][1].keys())))
# # print(Commutativity(str(*anti_commuting_sets[0][0].keys()), str(*anti_commuting_sets[2][1].keys())))
#



def Get_longest_anti_commuting_set(anti_commuting_sets):
    """
     Finds length of largest list in dictionary and returns it. In this case it will
     return the length of the largest anti_commuting set.

    Args:
        anti_commuting_sets (dic): Dictionary of anti_commuting sets of PauliWords

    Returns:
        (int): Length of largest anti_commuting set
    """
    max_key, max_value = max(anti_commuting_sets.items(), key=lambda x: len(x[1]))
    return len(max_value)

def Make_anti_commuting_sets_same_length(anti_commuting_sets, max_set_size):
    """
     Makes every set in anti_commuting_sets dictionary the same length, by adding None terms
     to instance when too short.

     This is very useful for finding 'longest tree'

    Args:
        anti_commuting_sets (dic): Dictionary of anti_commuting sets of PauliWords
        max_set_size (int): Length of longest anti_commuting set

    Returns:
        FILLED_anti_commuting_sets (dic): Dictionary of anti_commuting_sets with all sets the same size
    """

    FILLED_anti_commuting_sets = {}

    for key in anti_commuting_sets:
        len_term = len(anti_commuting_sets[key])
        if len_term == max_set_size:
            FILLED_anti_commuting_sets[key] = anti_commuting_sets[key]
        else:
            number_to_add = max_set_size - len_term
            None_list = [None for _ in range(number_to_add)]
            FILLED_anti_commuting_sets[key] = [*anti_commuting_sets[key], *None_list]
    return FILLED_anti_commuting_sets

max_set_size = Get_longest_anti_commuting_set(anti_commuting_sets)
FILLED_anti_commuting_sets = Make_anti_commuting_sets_same_length(anti_commuting_sets, max_set_size)


def Find_Longest_tree(FILLED_anti_commuting_sets, max_set_size, anti_comm=False):

    tree = {}
    best_reduction_possible = len(FILLED_anti_commuting_sets)  # <-- aka need this many things in branch_instance!
    running_best = 0

    for key in tqdm(range(len(FILLED_anti_commuting_sets)), ascii=True, desc='Getting best Branch'):
    #for key in FILLED_anti_commuting_sets:
        selected_set = FILLED_anti_commuting_sets[key]
        full_branch_key = []

        for i in range(len(selected_set)):
            P_word = selected_set[i] # this will be the 'top' of the tree
            branch_instance = []
            branch_instance_holder = {}

            if P_word is None:
                continue
            else:
                branch_instance.append(str(*P_word.keys()))
                jk_list = []
                for j in range(max_set_size):  # stays at 0 for all keys then goes up by 1 and repeats!

                    k_max = len(FILLED_anti_commuting_sets) - (key+1) # max number of levels one can loop through

                    for k in np.arange(key + 1, len(FILLED_anti_commuting_sets), 1):  # goes over different keys bellow top key
                        P_comp = FILLED_anti_commuting_sets[k][j]

                        if P_comp is None:
                            k_max -= 1
                            continue
                        else:
                            if False not in [Commutativity(term, str(*P_comp.keys()), anti_comm=anti_comm) for term in branch_instance]:
                                # print(key, i, k, j)
                                k_max -= 1
                                jk_list.append((j, k))
                                branch_instance.append(str(*P_comp.keys()))

                    if len(branch_instance) + k_max >= running_best:
                        continue
                    else:
                        ## print(branch_instance, '## VS ##','best: ', running_best, 'remaining: ', k_max)
                        break

                if running_best == best_reduction_possible:
                    break
                elif running_best < len(branch_instance):
                    running_best = len(branch_instance)
                    best_combo = {'key': key, 'i': i, 'j_k': jk_list}  # 'Branch_instance': branch_instance}

            branch_instance_holder.update({'i': i, 'j_k': jk_list, 'Branch_instance': branch_instance})
            full_branch_key.append(branch_instance_holder)

            if running_best == best_reduction_possible:
                break
        tree[key] = full_branch_key
        if running_best == best_reduction_possible:
            print('GET IN!!!')
            break
    return tree, best_combo

# actually only need best combo output!
# best combo defines which terms to make pauli S ! maximises commutativity!!!
tree_commute, best_combo_commute = Find_Longest_tree(FILLED_anti_commuting_sets, max_set_size, anti_comm= False)
#tree_anti, best_combo_anti = Find_Longest_tree(FILLED_anti_commuting_sets, max_set_size, anti_comm= True)


# want to maximise the anti_commutativity between the remaining terms to do further Unitary partitioning!
missing_k = [k for k in anti_commuting_sets.keys() if k not in [key for index, key in best_combo_commute['j_k']]]

# only look at missing keys!!!
new_anti_commuting_sets={}
i=0
for key in missing_k:
    # new_anti_commuting_sets[key]=anti_commuting_sets[key]
    new_anti_commuting_sets[i] = anti_commuting_sets[key]
    i+=1

max_set_size = Get_longest_anti_commuting_set(new_anti_commuting_sets)
NEW_FILLED_anti_commuting_sets = Make_anti_commuting_sets_same_length(new_anti_commuting_sets, max_set_size)

# HERE can either look for next best commutative tree OR look for best anti_commuting terms to do further UP too!
tree_anti, best_combo_anti = Find_Longest_tree(NEW_FILLED_anti_commuting_sets, max_set_size, anti_comm= True)

print(best_combo_anti)

# old idea (not working)
# # anti_commuting_sets key ordering... smallest first!
# dict_ordering_list = sorted(anti_commuting_sets, key=lambda k: len(anti_commuting_sets[k]), reverse=False)
# node_list = list(G.nodes)

# H = G.copy()
# #remove all of first set
# H.remove_nodes_from([P_word for sets in anti_commuting_sets[0] for P_word in sets])
#
# #remove all of second set BAR first term
# H.remove_nodes_from([P_word for sets in anti_commuting_sets[1][1::] for P_word in sets])
#
#
#
# # may be able to use max clique function (finds max no. of fully connected subgraphs!)
# # iterate through tree, selecting one P_word from each row
# # find max cliques
# list(nx.clique.find_cliques(H))
# min(list(nx.clique.find_cliques(H)))


####################

# # anti_commuting_sets key ordering... smallest first!
# dict_ordering_list = sorted(anti_commuting_sets, key=lambda k: len(anti_commuting_sets[k]), reverse=False)
#
# # get term with largest number of P_words
# max_num_terms = len(anti_commuting_sets[dict_ordering_list[-1]])

# LONGEST ANTI_COMMUTING SET