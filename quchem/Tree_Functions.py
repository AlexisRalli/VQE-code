from tqdm import tqdm
from quchem.Graph import Commutativity
from functools import reduce

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

                    # k_max = len(FILLED_anti_commuting_sets) - (key+1) # max number of levels one can loop through
                    k_max = len(key_list[INDEX + 1:])  # max number of levels one can loop through

                    # for k in np.arange(key + 1, len(FILLED_anti_commuting_sets), 1):  # goes over different keys bellow top key
                    for k in key_list[INDEX+1:]:
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


def Get_all_S_terms(anti_commuting_sets, anti_comm):

    S_dict={}
    grouped_terms =[]

    while len(anti_commuting_sets) >1:
        max_set_size = Get_longest_anti_commuting_set(anti_commuting_sets)
        FILLED_anti_commuting_sets = Make_anti_commuting_sets_same_length(anti_commuting_sets, max_set_size)

        tree, best_combo = Find_Longest_tree(FILLED_anti_commuting_sets, max_set_size, anti_comm)

        grouped_terms.append([(best_combo['i_key'][1], best_combo['i_key'][0])] + [(key, index) for index, key in best_combo['j_k']])


        S_dict[best_combo['i_key'][1]] = best_combo['i_key'][0]
        for index, key in best_combo['j_k']:  # , :
            S_dict[key] = index

        new_anti_commuting_sets = Remaining_anti_commuting_sets(best_combo, anti_commuting_sets)
        max_set_size = Get_longest_anti_commuting_set(new_anti_commuting_sets)

        anti_commuting_sets = Make_anti_commuting_sets_same_length(new_anti_commuting_sets, max_set_size)


    S_dict[list(anti_commuting_sets.keys())[0]] = 0 #TODO not the best way to find final term!!! (problem need to choose best term to reduce too when only have one term remaining!)
                                                    #TODO currently set to index 0... maybe look for most connected term here to all terms!
    grouped_terms.append([(list(anti_commuting_sets.keys())[0], 0)]) #TODO

    if reduce(lambda count, LIST: count + len(LIST), grouped_terms, 0) != len(S_dict):
        raise ValueError('incorrect grouped terms')


    return S_dict, grouped_terms





# vv = {34: [{'I0 Y1 Z2 Z3 Z4 Y5 I6 I7 Z8 I9 I10 I11': {'Cofactors': (-0.0010954854746108847+0j)}}, {'I0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 I8 Z9 Z10 Y11': {'Cofactors': (-0.000818652430590076+0j)}}, {'I0 Y1 Z2 Y3 I4 I5 I6 I7 I8 Z9 I10 I11': {'Cofactors': (-0.0035789393681122253+0j)}}, {'Y0 I1 Y2 I3 I4 I5 I6 I7 I8 I9 I10 I11': {'Cofactors': (-0.04069607268303362+0j)}}, {'Y0 I1 Z2 Z3 Y4 I5 I6 I7 I8 I9 I10 I11': {'Cofactors': (-0.031472346526068146+0j)}}, {'Y0 I1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 I11': {'Cofactors': (-0.017263560047055176+0j)}}, None, None, None], 41: [{'I0 I1 X2 Z3 X4 I5 I6 I7 Z8 I9 I10 I11': {'Cofactors': (0.004685763115344226+0j)}}, {'I0 I1 X2 Z3 Z4 Z5 Z6 Z7 I8 Z9 X10 I11': {'Cofactors': (-0.009767493327321967+0j)}}, {'Y0 Z1 I2 Z3 Y4 I5 I6 I7 I8 I9 I10 I11': {'Cofactors': (-0.00479117047502374+0j)}}, {'Y0 Z1 I2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 I11': {'Cofactors': (-0.0017809703751826705+0j)}}, {'I0 I1 X2 Y3 I4 I5 Y6 X7 I8 I9 I10 I11': {'Cofactors': (0.006795528976406602+0j)}}, {'I0 I1 X2 Z3 Z4 Y5 Y6 X7 I8 I9 I10 I11': {'Cofactors': (0.004889539651547371+0j)}}, {'I0 I1 X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11': {'Cofactors': (-0.004031651676556206+0j)}}, None, None], 42: [{'I0 I1 Y2 Z3 Y4 I5 I6 I7 I8 Z9 I10 I11': {'Cofactors': (-0.00020377653620314614+0j)}}, {'I0 I1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 I9 Y10 I11': {'Cofactors': (-0.00573584165076576+0j)}}, {'Y0 Z1 Z2 Z3 I4 Z5 Z6 Z7 Z8 Z9 Y10 I11': {'Cofactors': (-0.0013724367918927529+0j)}}, {'Y0 Z1 Z2 Z3 Y4 I5 I6 I7 I8 I9 Z10 I11': {'Cofactors': (-0.003634075682277503+0j)}}, None, None, None, None, None], 43: [{'I0 I1 X2 Z3 X4 I5 I6 I7 I8 Z9 I10 I11': {'Cofactors': (-0.00020377653620314614+0j)}}, {'I0 I1 X2 Z3 Z4 Z5 Z6 Z7 Z8 I9 X10 I11': {'Cofactors': (-0.00573584165076576+0j)}}, {'X0 Z1 Z2 Z3 I4 Z5 Z6 Z7 Z8 Z9 X10 I11': {'Cofactors': (-0.0013724367918927529+0j)}}, {'X0 Z1 Z2 Z3 X4 I5 I6 I7 I8 I9 Z10 I11': {'Cofactors': (-0.003634075682277503+0j)}}, None, None, None, None, None], 48: [{'I0 I1 I2 X3 Z4 X5 I6 I7 Z8 I9 I10 I11': {'Cofactors': (-0.00020377653620314614+0j)}}, {'I0 I1 I2 X3 Z4 Z5 Z6 Z7 I8 Z9 Z10 X11': {'Cofactors': (-0.00573584165076576+0j)}}, {'I0 Y1 Z2 I3 Z4 Y5 I6 I7 I8 I9 I10 I11': {'Cofactors': (-0.00479117047502374+0j)}}, {'I0 Y1 Z2 I3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11': {'Cofactors': (-0.0017809703751826705+0j)}}, {'I0 I1 I2 X3 Y4 I5 Y6 X7 I8 I9 I10 I11': {'Cofactors': (-0.004889539651547371+0j)}}, {'I0 I1 I2 X3 Z4 Z5 X6 Y7 Z8 Z9 Y10 I11': {'Cofactors': (0.004031651676556206+0j)}}, None, None, None], 49: [{'I0 I1 I2 Y3 Z4 Y5 I6 I7 I8 Z9 I10 I11': {'Cofactors': (0.004685763115344226+0j)}}, {'I0 I1 I2 Y3 Z4 Z5 Z6 Z7 Z8 I9 Z10 Y11': {'Cofactors': (-0.009767493327321967+0j)}}, {'I0 Y1 Z2 Z3 Z4 I5 Z6 Z7 Z8 Z9 Z10 Y11': {'Cofactors': (-0.0013724367918927529+0j)}}, {'I0 Y1 Z2 Z3 Z4 Y5 I6 I7 I8 I9 I10 Z11': {'Cofactors': (-0.003634075682277503+0j)}}, None, None, None, None, None], 50: [{'I0 I1 I2 X3 Z4 X5 I6 I7 I8 Z9 I10 I11': {'Cofactors': (0.004685763115344226+0j)}}, {'I0 I1 I2 X3 Z4 Z5 Z6 Z7 Z8 I9 Z10 X11': {'Cofactors': (-0.009767493327321967+0j)}}, {'I0 X1 Z2 Z3 Z4 I5 Z6 Z7 Z8 Z9 Z10 X11': {'Cofactors': (-0.0013724367918927529+0j)}}, {'I0 X1 Z2 Z3 Z4 X5 I6 I7 I8 I9 I10 Z11': {'Cofactors': (-0.003634075682277503+0j)}}, None, None, None, None, None], 67: [{'Y0 Z1 Y2 I3 Y4 Z5 Z6 Z7 Z8 Z9 Y10 I11': {'Cofactors': (0.0017145480548576523+0j)}}, {'Z0 I1 I2 I3 Y4 Z5 Z6 Z7 Z8 Z9 Y10 I11': {'Cofactors': (-0.002970729104739796+0j)}}, {'I0 I1 I2 I3 I4 I5 Y6 X7 I8 I9 X10 Y11': {'Cofactors': (0.0038329864095138673+0j)}}, {'Y0 X1 I2 I3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11': {'Cofactors': (0.0022963158525226954+0j)}}, {'Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11': {'Cofactors': (0.003240390663727458+0j)}}, {'Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11': {'Cofactors': (-0.0012919456388330141+0j)}}, {'Y0 Z1 Z2 Z3 Y4 I5 I6 I7 I8 I9 I10 Z11': {'Cofactors': (-0.004030774942548389+0j)}}, None, None], 74: [{'I0 Y1 Z2 Y3 I4 X5 Z6 Z7 Z8 Z9 Z10 X11': {'Cofactors': (-0.0004976272121178758+0j)}}, {'I0 Z1 I2 I3 I4 X5 Z6 Z7 Z8 Z9 Z10 X11': {'Cofactors': (-0.002970729104739796+0j)}}, {'Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11': {'Cofactors': (-0.017263560047055176+0j)}}, {'Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11': {'Cofactors': (0.002746861819556288+0j)}}, {'Y0 Z1 Z2 X3 X4 Y5 I6 I7 I8 I9 I10 I11': {'Cofactors': (-0.00015508053970906706+0j)}}, {'Y0 Z1 Z2 X3 I4 X5 Z6 Z7 Z8 Z9 Y10 I11': {'Cofactors': (-0.0010282153967519291+0j)}}, None, None, None], 75: [{'I0 X1 Z2 X3 I4 Y5 Z6 Z7 Z8 Z9 Z10 Y11': {'Cofactors': (-0.0004976272121178758+0j)}}, {'I0 Y1 Z2 Y3 I4 Z5 I6 I7 I8 I9 I10 I11': {'Cofactors': (-0.004230658824091613+0j)}}, {'X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11': {'Cofactors': (-0.002746861819556288+0j)}}, {'X0 Z1 Z2 I3 X4 I5 I6 I7 I8 I9 I10 I11': {'Cofactors': (-0.00642657703520624+0j)}}, {'X0 Z1 Z2 I3 Z4 Z5 Z6 Z7 Z8 Z9 X10 I11': {'Cofactors': (0.0013559719031550977+0j)}}, {'I0 I1 Z2 Y3 Z4 Y5 I6 I7 I8 I9 I10 I11': {'Cofactors': (0.009702964590343708+0j)}}, {'I0 I1 Y2 X3 X4 Y5 I6 I7 I8 I9 I10 I11': {'Cofactors': (0.002366482760463115+0j)}}, {'I0 I1 Y2 X3 I4 X5 Z6 Z7 Z8 Z9 Y10 I11': {'Cofactors': (0.0072371008531110755+0j)}}, None], 83: [{'Y0 Y1 I2 I3 I4 I5 I6 I7 I8 I9 X10 X11': {'Cofactors': (-0.001774434979349804+0j)}}, {'Y0 Z1 Z2 I3 Y4 I5 I6 I7 I8 I9 I10 I11': {'Cofactors': (-0.00642657703520624+0j)}}, {'Y0 Z1 Z2 Y3 X4 X5 I6 I7 I8 I9 I10 I11': {'Cofactors': (0.00015508053970906706+0j)}}, {'I0 Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 I11': {'Cofactors': (-0.0031369422783377675+0j)}}, {'I0 Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10 I11': {'Cofactors': (0.0027427634516095827+0j)}}, {'I0 I1 X2 Y3 I4 I5 I6 I7 I8 I9 Y10 X11': {'Cofactors': (0.03060390781148419+0j)}}, None, None, None], 84: [{'Y0 Z1 Z2 I3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 I11': {'Cofactors': (0.0013559719031550977+0j)}}, {'X0 Z1 X2 X3 Z4 X5 I6 I7 I8 I9 I10 I11': {'Cofactors': (0.0016354065601825016+0j)}}, {'X0 Z1 Z2 X3 Y4 Y5 I6 I7 I8 I9 I10 I11': {'Cofactors': (0.00015508053970906706+0j)}}, {'Y0 Z1 Y2 I3 I4 Z5 I6 I7 I8 I9 I10 I11': {'Cofactors': (-0.004075578284382545+0j)}}, {'Y0 Z1 Z2 Z3 Y4 Z5 I6 I7 I8 I9 I10 I11': {'Cofactors': (0.000814468773457972+0j)}}, {'I0 Y1 X2 I3 I4 X5 Z6 Z7 Z8 Z9 Y10 I11': {'Cofactors': (0.003240390663727458+0j)}}, {'I0 I1 Y2 Z3 Z4 Y5 I6 I7 I8 I9 X10 X11': {'Cofactors': (0.007246981464973575+0j)}}, None, None], 85: [{'Z0 I1 I2 Y3 Z4 Y5 I6 I7 I8 I9 I10 I11': {'Cofactors': (-0.0004874738967788586+0j)}}, {'Y0 Z1 Y2 Y3 Z4 Y5 I6 I7 I8 I9 I10 I11': {'Cofactors': (0.0016354065601825016+0j)}}, {'Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11': {'Cofactors': (-0.003240390663727458+0j)}}, {'Y0 Z1 Z2 Z3 Z4 X5 I6 I7 I8 I9 X10 Y11': {'Cofactors': (0.0003966992602708843+0j)}}, {'I0 I1 Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11': {'Cofactors': (-0.007237100853111076+0j)}}, {'I0 I1 I2 Y3 I4 Z5 Z6 Z7 Z8 Z9 Z10 Y11': {'Cofactors': (-0.0038464821371521434+0j)}}, {'I0 I1 I2 Y3 Y4 I5 I6 I7 I8 I9 Y10 Y11': {'Cofactors': (0.007246981464973575+0j)}}, None, None], 86: [{'X0 Z1 Z2 Y3 Y4 X5 I6 I7 I8 I9 I10 I11': {'Cofactors': (-0.00015508053970906706+0j)}}, {'X0 Z1 Z2 Z3 X4 Z5 I6 I7 I8 I9 I10 I11': {'Cofactors': (0.000814468773457972+0j)}}, {'Y0 Z1 Y2 I3 I4 Y5 Z6 Z7 Z8 Z9 Z10 Y11': {'Cofactors': (0.0027427634516095827+0j)}}, {'X0 Z1 X2 I3 I4 I5 I6 I7 I8 I9 I10 Z11': {'Cofactors': (0.0037160401092958665+0j)}}, {'X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11': {'Cofactors': (0.002519150807643293+0j)}}, {'I0 Y1 Z2 Z3 I4 Z5 Z6 Z7 Z8 Z9 Z10 Y11': {'Cofactors': (-8.04911530597382e-05+0j)}}, {'I0 Y1 Z2 Z3 X4 I5 I6 I7 I8 I9 X10 Y11': {'Cofactors': (-0.0003966992602708843+0j)}}, None, None], 88: [{'X0 Z1 X2 I3 I4 Z5 I6 I7 I8 I9 I10 I11': {'Cofactors': (-0.004075578284382545+0j)}}, {'X0 Z1 Z2 Z3 Z4 I5 Z6 Z7 Z8 Z9 X10 I11': {'Cofactors': (-8.04911530597382e-05+0j)}}, {'Y0 Z1 Y2 I3 I4 X5 Z6 Z7 Z8 Z9 Z10 X11': {'Cofactors': (0.0027427634516095827+0j)}}, {'Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11': {'Cofactors': (-0.0012919456388330141+0j)}}, {'X0 Z1 Z2 Z3 X4 I5 I6 I7 I8 I9 I10 Z11': {'Cofactors': (-0.004030774942548389+0j)}}, {'I0 Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10 I11': {'Cofactors': (-0.0012919456388330143+0j)}}, {'I0 I1 I2 Y3 X4 X5 Z6 Z7 Z8 Z9 Y10 I11': {'Cofactors': (0.0012091978607673857+0j)}}, None, None], 89: [{'Z0 I1 I2 I3 I4 X5 Z6 Z7 Z8 Z9 Z10 X11': {'Cofactors': (-0.005267044957262492+0j)}}, {'X0 Z1 X2 I3 I4 Y5 Z6 Z7 Z8 Z9 Z10 Y11': {'Cofactors': (0.0027427634516095827+0j)}}, {'X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11': {'Cofactors': (-0.0012919456388330141+0j)}}, {'X0 Z1 Z2 Z3 Z4 Y5 I6 I7 I8 I9 Y10 X11': {'Cofactors': (0.0003966992602708843+0j)}}, None, None, None, None, None], 90: [{'X0 Z1 X2 I3 I4 X5 Z6 Z7 Z8 Z9 Z10 X11': {'Cofactors': (0.0027427634516095827+0j)}}, {'X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11': {'Cofactors': (-0.0012919456388330141+0j)}}, {'I0 X1 Z2 Z3 Y4 I5 I6 I7 I8 I9 Y10 X11': {'Cofactors': (-0.0003966992602708843+0j)}}, {'I0 Y1 Z2 Z3 Z4 Y5 I6 I7 I8 I9 Z10 I11': {'Cofactors': (-0.004030774942548389+0j)}}, {'I0 I1 I2 X3 Y4 I5 I6 I7 I8 I9 Y10 X11': {'Cofactors': (0.007246981464973575+0j)}}, {'I0 I1 I2 Y3 Z4 Y5 I6 I7 I8 I9 Z10 I11': {'Cofactors': (0.009032995434074152+0j)}}, {'I0 I1 I2 I3 X4 Z5 Z6 Z7 Z8 Z9 X10 Z11': {'Cofactors': (0.009965850807740591+0j)}}, None, None], 91: [{'I0 Y1 I2 Y3 I4 I5 I6 I7 I8 I9 I10 I11': {'Cofactors': (0.003714482527051009+0j)}}, {'I0 Y1 X2 X3 Y4 I5 I6 I7 I8 I9 I10 I11': {'Cofactors': (0.0016354065601825016+0j)}}, {'I0 X1 X2 I3 I4 Y5 Z6 Z7 Z8 Z9 Y10 I11': {'Cofactors': (-0.003240390663727458+0j)}}, {'I0 X1 X2 I3 I4 I5 I6 I7 I8 I9 X10 X11': {'Cofactors': (0.002102865763334828+0j)}}, {'I0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 I10 Y11': {'Cofactors': (0.002519150807643293+0j)}}, {'I0 I1 I2 Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10 I11': {'Cofactors': (0.01214458053073908+0j)}}, None, None, None], 92: [{'I0 X1 I2 Z3 Z4 X5 I6 I7 I8 I9 I10 I11': {'Cofactors': (-0.00642657703520624+0j)}}, {'I0 Y1 Y2 X3 X4 I5 I6 I7 I8 I9 I10 I11': {'Cofactors': (-0.0016354065601825016+0j)}}, {'I0 Y1 X2 I3 X4 Y5 I6 I7 I8 I9 I10 I11': {'Cofactors': (0.00015508053970906708+0j)}}, {'I0 Y1 X2 I3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11': {'Cofactors': (-0.0010282153967519291+0j)}}, {'I0 Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 I11': {'Cofactors': (0.0031369422783377675+0j)}}, {'I0 I1 Y2 Y3 I4 I5 I6 I7 I8 I9 X10 X11': {'Cofactors': (-0.03060390781148419+0j)}}, None, None, None], 93: [{'I0 X1 X2 I3 X4 X5 I6 I7 I8 I9 I10 I11': {'Cofactors': (0.00015508053970906708+0j)}}, {'I0 X1 X2 I3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11': {'Cofactors': (-0.0010282153967519291+0j)}}, {'I0 Y1 Z2 Y3 Z4 I5 I6 I7 I8 I9 I10 I11': {'Cofactors': (-0.004075578284382545+0j)}}, {'I0 Y1 Z2 Z3 I4 Y5 I6 I7 I8 I9 I10 I11': {'Cofactors': (0.000814468773457972+0j)}}, None, None, None, None, None], 95: [{'I0 Y1 Y2 I3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11': {'Cofactors': (-0.0010282153967519291+0j)}}, {'I0 Y1 Y2 I3 I4 I5 I6 I7 I8 I9 Y10 Y11': {'Cofactors': (0.002102865763334828+0j)}}, {'I0 I1 Z2 X3 Z4 X5 I6 I7 I8 I9 I10 I11': {'Cofactors': (0.009702964590343708+0j)}}, {'I0 I1 Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11': {'Cofactors': (-0.03998383732969973+0j)}}, {'I0 I1 Y2 X3 I4 I5 I6 I7 I8 I9 X10 Y11': {'Cofactors': (0.03060390781148419+0j)}}, None, None, None, None], 96: [{'I0 X1 Z2 Z3 I4 X5 I6 I7 I8 I9 I10 I11': {'Cofactors': (0.000814468773457972+0j)}}, {'I0 X1 Z2 Z3 I4 Z5 Z6 Z7 Z8 Z9 Z10 X11': {'Cofactors': (-8.04911530597382e-05+0j)}}, {'I0 X1 Z2 X3 I4 I5 I6 I7 I8 I9 Z10 I11': {'Cofactors': (0.0037160401092958665+0j)}}, {'I0 I1 X2 I3 X4 I5 I6 I7 I8 I9 I10 I11': {'Cofactors': (0.00970296459034371+0j)}}, {'I0 I1 X2 X3 Y4 Y5 I6 I7 I8 I9 I10 I11': {'Cofactors': (-0.002366482760463115+0j)}}, {'I0 I1 X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11': {'Cofactors': (0.007237100853111076+0j)}}, None, None, None], 97: [{'I0 X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10 I11': {'Cofactors': (0.0027427634516095827+0j)}}, {'I0 X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10 I11': {'Cofactors': (-0.0012919456388330143+0j)}}, {'I0 I1 Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11': {'Cofactors': (0.007237100853111076+0j)}}, {'I0 I1 Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11': {'Cofactors': (0.0012091978607673857+0j)}}, {'I0 I1 Y2 Z3 Y4 I5 I6 I7 I8 I9 I10 Z11': {'Cofactors': (0.009032995434074152+0j)}}, None, None, None, None], 98: [{'I0 X1 Z2 Z3 Z4 X5 I6 I7 I8 I9 Z10 I11': {'Cofactors': (-0.004030774942548389+0j)}}, {'I0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 I10 X11': {'Cofactors': (0.002519150807643293+0j)}}, {'I0 I1 X2 I3 Z4 Z5 Z6 Z7 Z8 Z9 X10 I11': {'Cofactors': (-0.03998383732969973+0j)}}, {'I0 I1 X2 X3 I4 X5 Z6 Z7 Z8 Z9 X10 I11': {'Cofactors': (0.0072371008531110755+0j)}}, {'I0 I1 X2 X3 I4 I5 I6 I7 I8 I9 Y10 Y11': {'Cofactors': (-0.03060390781148419+0j)}}, None, None, None, None], 99: [{'I0 I1 X2 Y3 Y4 X5 I6 I7 I8 I9 I10 I11': {'Cofactors': (0.002366482760463115+0j)}}, {'I0 I1 X2 Y3 I4 Y5 Z6 Z7 Z8 Z9 X10 I11': {'Cofactors': (0.0072371008531110755+0j)}}, {'I0 I1 Z2 I3 I4 X5 Z6 Z7 Z8 Z9 Z10 X11': {'Cofactors': (0.01214458053073908+0j)}}, {'I0 I1 Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11': {'Cofactors': (0.0012091978607673857+0j)}}, None, None, None, None, None], 100: [{'I0 I1 Y2 Y3 I4 Y5 Z6 Z7 Z8 Z9 Y10 I11': {'Cofactors': (0.0072371008531110755+0j)}}, {'I0 I1 X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11': {'Cofactors': (-0.007237100853111076+0j)}}, {'I0 I1 X2 Z3 Z4 I5 Z6 Z7 Z8 Z9 X10 I11': {'Cofactors': (-0.0038464821371521434+0j)}}, {'I0 I1 X2 Z3 Z4 X5 I6 I7 I8 I9 Y10 Y11': {'Cofactors': (0.007246981464973575+0j)}}, {'I0 I1 I2 Z3 X4 Z5 Z6 Z7 Z8 Z9 X10 I11': {'Cofactors': (0.01214458053073908+0j)}}, None, None, None, None]}
# max_set_size = Get_longest_anti_commuting_set(vv)
# FILLED_anti_commuting_sets = Make_anti_commuting_sets_same_length(vv, max_set_size)
# tree, best_combo = Find_Longest_tree(FILLED_anti_commuting_sets, max_set_size, 'C')









# def Get_PauliS_terms_from_tree(anti_commuting_sets, *List_best_combo):
#     S_dict = {}
#
#     for best_combo in List_best_combo:
#         S_dict[best_combo['i_key'][1]] = best_combo['i_key'][0]
#         for index, key in best_combo['j_k']:  # , :
#             S_dict[key] = index
#
#     return S_dict

