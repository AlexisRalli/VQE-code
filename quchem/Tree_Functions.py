from tqdm import tqdm
from quchem.Graph import Commutativity

def Get_longest_anti_commuting_set(anti_commuting_sets):
    """
     Finds length of largest list in dictionary and returns it. In this case it will
     return the length of the largest anti_commuting set.

    Args:
        anti_commuting_sets (dict): Dictionary of anti_commuting sets of PauliWords

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
        anti_commuting_sets (dict): Dictionary of anti_commuting sets of PauliWords
        max_set_size (int): Length of longest anti_commuting set

    Returns:
        FILLED_anti_commuting_sets (dic): Dictionary of anti_commuting_sets with all sets the same size

    .. code-block:: python
   :emphasize-lines: 15

   from quchem.Tree_Fucntions import *

   anti_commuting_sets = {
                            0: [{'I0 I1 I2 I3 I4 I5 I6 I7 I8 I9 I10 I11': {'Cofactors': (-3.9344419569678517+0j)}}],

                            1: [ {'I0 I1 I2 I3 I4 I5 Z6 I7 I8 I9 I10 I11': {'Cofactors': (-0.21675429325000556+0j)}},
                                 {'I0 I1 I2 I3 I4 I5 Y6 X7 X8 Y9 I10 I11': {'Cofactors': (0.004217284878422758+0j)}},
                                 {'Y0 Y1 I2 I3 I4 I5 X6 X7 I8 I9 I10 I11': {'Cofactors': (-0.002472706153881531+0j)}},
                                 {'Y0 Z1 Z2 Y3 I4 I5 X6 X7 I8 I9 I10 I11': {'Cofactors': (0.0020778874983957123+0j)}},
                                 {'Y0 Z1 Z2 Z3 Z4 Y5 X6 X7 I8 I9 I10 I11': {'Cofactors': (0.002562389780011484+0j)}},
                                 {'Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11': {'Cofactors': (0.0009084689616229714+0j)}}
                               ]
                         }
   max_set_size = Get_longest_anti_commuting_set(anti_commuting_sets)
   Make_anti_commuting_sets_same_length(anti_commuting_sets, max_set_size)
   >> {0: [ {'I0 I1 I2 I3 I4 I5 I6 I7 I8 I9 I10 I11': {'Cofactors': (-3.9344419569678517+0j)}},
              None,
              None,
              None,
              None,
              None
           ],

     1:  [ {'I0 I1 I2 I3 I4 I5 Z6 I7 I8 I9 I10 I11': {'Cofactors': (-0.21675429325000556+0j)}},
           {'I0 I1 I2 I3 I4 I5 Y6 X7 X8 Y9 I10 I11': {'Cofactors': (0.004217284878422758+0j)}},
           {'Y0 Y1 I2 I3 I4 I5 X6 X7 I8 I9 I10 I11': {'Cofactors': (-0.002472706153881531+0j)}},
           {'Y0 Z1 Z2 Y3 I4 I5 X6 X7 I8 I9 I10 I11': {'Cofactors': (0.0020778874983957123+0j)}},
           {'Y0 Z1 Z2 Z3 Z4 Y5 X6 X7 I8 I9 I10 I11': {'Cofactors': (0.002562389780011484+0j)}},
           {'Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11': {'Cofactors': (0.0009084689616229714+0j)}}
         ]
       }

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

def Find_Longest_tree(FILLED_anti_commuting_sets, max_set_size, anti_comm):
    """
     From a dictionary of fully anti-commuting sets (each entry is of same length, None used as filler. Finds
     longests possible either fully anti-commuting or fully commuting subsets between rows #TODO need to find QWC too!


    Args:
        FILLED_anti_commuting_sets (dict): Dictionary of anti_commuting sets of PauliWords
        max_set_size (int): Length of longest anti_commuting set
        anti_comm_QWC (str): flags to find either:
                                           qubit wise commuting (QWC) terms  -> flag = 'QWC',
                                                             commuting terms -> flag = 'C',
                                                        anti-commuting terms -> flag = 'AC'

    Returns:
        tree (dict): Dictionary of selected PauliWord: i_key = (index, row) and j_k = list of (index, row) either
                    commuting or anti-commuting PauliWords (list)
        best_combo (dict): selected PauliWord: i_key = (index, row) and j_k = list of (index, row) of best possible
                          sub set

    """
    tree = {}
    best_reduction_possible = len(FILLED_anti_commuting_sets)  # <-- aka how many fully anti-commuting sets
    running_best = 0

    key_list = list(FILLED_anti_commuting_sets.keys())

    #for key in FILLED_anti_commuting_sets:
    for INDEX in tqdm(range(len(FILLED_anti_commuting_sets)), ascii=True, desc='Getting best Branch'):
        key = key_list[INDEX]
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

                    # for k in np.arange(key + 1, len(FILLED_anti_commuting_sets), 1):  # goes over different keys bellow top key
                    for k in key_list[key+1:]:
                        P_comp = FILLED_anti_commuting_sets[k][j]

                        if P_comp is None:
                            k_max -= 1
                            continue
                        else:
                            if False not in [Commutativity(term, str(*P_comp.keys()), anti_comm) for term in branch_instance]:
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
                    best_combo = {'i_key': (i, key), 'j_k': jk_list}  # 'Branch_instance': branch_instance}

            branch_instance_holder.update({'i_key': (i, key), 'j_k': jk_list, 'Branch_instance': branch_instance})
            full_branch_key.append(branch_instance_holder)

            if running_best == best_reduction_possible:
                break
        tree[key] = full_branch_key
        if running_best == best_reduction_possible:
            print('GET IN!!!')
            break
    return tree, best_combo

def Remaining_anti_commuting_sets(best_combo, anti_commuting_sets):
    """
     Gives Remaining terms in anti commuting sets that are NOT present in best_combo (aka sub set that either
     fully commutes / anti_commutes.
     Can perform further analysis on new dictionary (by repeated use of Find Longest Tree function.


    Args:
        best_combo (dict): selected PauliWord: i_key = (index, row) and j_k = list of (index, row) of best possible
                          sub set
        anti_commuting_sets (dict): Dictionary of anti_commuting sets of PauliWords


    Returns:
        new_anti_commuting_sets (dict): returns dictionary of anti_commuting_sets but with rows removed if present in
                                       best_combo (dict)


    """
    missing_k = [k for k in anti_commuting_sets.keys() if
                 k not in [key for index, key in best_combo['j_k']] + [best_combo['i_key'][1]]]
    new_anti_commuting_sets = {}
    for key in missing_k:
        new_anti_commuting_sets[key] = anti_commuting_sets[key]
    return new_anti_commuting_sets