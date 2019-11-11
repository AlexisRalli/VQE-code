from Unitary_partitioning import *
import pytest
import numpy as np
# in Terminal run (LINUX!):
# py.test /home/alexisr/Documents/PhD/Code/PhD\ Code/test_Unitary_partitioning.py -v


def test_Get_beta_j_cofactors_normalisation():
    '''
    Standard use case.

    Here checking if  (sum_j B_j^2 = 1) according to eq (10) in ArXiv:1908.08067

    :return:
    '''
    ANTI_commuting_sets = {
        0: [('Z0 I1 I2 I3', (0.1371657293179602 + 0j)), ('Y0 X1 X2 Y3', (0.04919764587885283 + 0j)),
            ('X0 I1 I2 I3', (0.04919764587885283 + 0j))],
        1: [('I0 Z1 I2 I3', (0.1371657293179602 + 0j)), ('Y0 Y1 X2 X3', (-0.04919764587885283 + 0j))],
        2: [('Z0 I1 Z2 I3', (0.10622904488350779 + 0j))]
    }

    updated_anti_commuting_sets = Get_beta_j_cofactors(ANTI_commuting_sets)

    summation_list=[]
    for key in updated_anti_commuting_sets:
        summation = sum([constant**2 for PauliWord, constant in updated_anti_commuting_sets[key]['PauliWords']])

        if not np.isclose(1+0j, summation, rtol=1e-09, atol=0.0):
            raise ValueError('B_j^2 terms sum too {} rather than 1'.format(summation))

        summation_list.append(True)

    assert all(summation_list) == True

def test_Get_beta_j_cofactors_constants():
    '''
    Standard use case.

    Makes sure that factor can be used to re-gain old constant!

    :return:
    '''
    ANTI_commuting_sets = {
        0: [('Z0 I1 I2 I3', (0.1371657293179602 + 0j)), ('Y0 X1 X2 Y3', (0.04919764587885283 + 0j)),
            ('X0 I1 I2 I3', (0.04919764587885283 + 0j))],
        1: [('I0 Z1 I2 I3', (0.1371657293179602 + 0j)), ('Y0 Y1 X2 X3', (-0.04919764587885283 + 0j))],
        2: [('Z0 I1 Z2 I3', (0.10622904488350779 + 0j))]
    }

    updated_anti_commuting_sets = Get_beta_j_cofactors(ANTI_commuting_sets)

    check_list=[]
    for key in updated_anti_commuting_sets:
        factor = updated_anti_commuting_sets[key]['factor']

        CORRECT_old_constants = [PauliWord_and_constant[1] for PauliWord_and_constant in ANTI_commuting_sets[key]] # ['PauliWords']]

        calculating_old_constants = [np.sqrt(factor)*constant for PauliWord, constant in updated_anti_commuting_sets[key]['PauliWords']]

        # NOTE this does NOT work if you do:
        # checking_old_constants = [np.sqrt(constant**2 *factor) for PauliWord, constant in updated_anti_commuting_sets[key]['PauliWords']]
        #                                      ^^^^^  note this part makes negative constants positive!!!

        check_list.append(CORRECT_old_constants==calculating_old_constants)

    assert all(check_list) == True

def test_Get_beta_j_cofactors_manual():
    '''
    Standard use case.
    Takes in anti-commuting sets and returns same sets, but with cofactors obeying (sum_j B_j^2 = 1)

    Function takes in anti_commuting_sets and returns anti-commuting sets, but with new coefcators that
    obey eq (10) in ArXiv:1908.08067 (sum_j B_j^2 = 1)

    :return:
    '''
    ANTI_commuting_sets = {
        0: [('Z0 I1 I2 I3', (0.1371657293179602 + 0j)), ('Y0 X1 X2 Y3', (0.04919764587885283 + 0j)),
            ('X0 I1 I2 I3', (0.04919764587885283 + 0j))],
        1: [('I0 Z1 I2 I3', (0.1371657293179602 + 0j)), ('Y0 Y1 X2 X3', (-0.04919764587885283 + 0j))],
        2: [('Z0 I1 Z2 I3', (0.10622904488350779 + 0j))]
    }

    CORRECT = {}
    for key, value in ANTI_commuting_sets.items():
        factor = sum([constant**2 for PauliWord, constant in value])

        terms = []
        for PauliWord, constant in value:
            new_constant = constant/np.sqrt(factor)
            terms.append((PauliWord, new_constant))

        CORRECT[key] = {'PauliWords': terms, 'factor': factor}

    for key in CORRECT:
        summation = sum([constant**2 for PauliWord, constant in CORRECT[key]['PauliWords']])

        if not np.isclose(1+0j, summation, rtol=1e-09, atol=0.0):
            raise ValueError('B_j^2 terms sum too {} rather than 1'.format(summation))

    updated_anti_commuting_sets = Get_beta_j_cofactors(ANTI_commuting_sets)

    assert CORRECT == updated_anti_commuting_sets

##
def test_Get_X_sk_operators():
    """
    Standard use case

    :return:
    """
    S = 0

    normalised_anti_commuting_sets ={
         0: {'PauliWords': [('I0 I1 Z2 Z3', (1 + 0j))],
             'factor': (0.02665633752583814 + 0j)},

         1: {'PauliWords': [('Z0 I1 I2 I3', (0.8918294488900189 + 0j)),
                            ('Y0 X1 X2 Y3', (0.3198751585326103 + 0j)),
                            ('X0 I1 I2 I3', (0.3198751585326103 + 0j))],
             'factor': (0.023655254019369937 + 0j)},

         2: {'PauliWords': [('I0 Z1 I2 I3', (0.9412848366792171 + 0j)),
                            ('Y0 Y1 X2 X3', (-0.33761347164735517 + 0j))],
             'factor': (0.021234845659348932 + 0j)},
        }

    X_sk_and_theta_sk = Get_X_sk_operators(normalised_anti_commuting_sets, S=S)

    check_X_sk_and_theta_sk={}
    for key in normalised_anti_commuting_sets:

        anti_commuting_set = normalised_anti_commuting_sets[key]['PauliWords']

        if len(anti_commuting_set) > 1:
            k_indexes = [index for index in range(len(anti_commuting_set)) if
                         index != S]

            Op_list = []
            for k in k_indexes:
                X_sk_op = (anti_commuting_set[S], anti_commuting_set[k])

                tan_theta_sk = anti_commuting_set[k][1] / (np.sqrt( anti_commuting_set[S][1] + sum([anti_commuting_set[beta_j][1]**2 for beta_j
                                                                                         in np.arange(1,k, 1)]))) #eqn 16

                theta_sk = np.arctan(tan_theta_sk)

                Op_list.append({'X_sk': X_sk_op, 'theta_sk': theta_sk, 'factor': normalised_anti_commuting_sets[key]['factor']})

                check_X_sk_and_theta_sk.update({key: Op_list})

    assert check_X_sk_and_theta_sk == X_sk_and_theta_sk

def test_Get_X_sk_operators_manual():
    """
    Standard use case

    :return:
    """
    S = 0

    normalised_anti_commuting_sets ={
         0: {'PauliWords': [('I0 I1 Z2 Z3', (1 + 0j))],
             'factor': (0.02665633752583814 + 0j)},

         1: {'PauliWords': [('Z0 I1 I2 I3', (0.8918294488900189 + 0j)),
                            ('Y0 X1 X2 Y3', (0.3198751585326103 + 0j)),
                            ('X0 I1 I2 I3', (0.3198751585326103 + 0j))],
             'factor': (0.023655254019369937 + 0j)},

         2: {'PauliWords': [('I0 Z1 I2 I3', (0.9412848366792171 + 0j)),
                            ('Y0 Y1 X2 X3', (-0.33761347164735517 + 0j))],
             'factor': (0.021234845659348932 + 0j)},
        }

    X_sk_and_theta_sk = Get_X_sk_operators(normalised_anti_commuting_sets, S=S)

    MANUAL_answer = {1:
                         [
                             {'X_sk': (('Z0 I1 I2 I3', (0.8918294488900189+0j)),
                             ('Y0 X1 X2 Y3', (0.3198751585326103+0j))),
                             'theta_sk': (FIND),
                             'factor': (0.023655254019369937+0j)},

                             {'X_sk': (('Z0 I1 I2 I3', (0.8918294488900189+0j)),
                             ('X0 I1 I2 I3', (0.3198751585326103+0j))),
                             'theta_sk': (FIND),
                              'factor': (0.023655254019369937+0j)}
                         ],

              
                2:
                    [
                        {'X_sk': (('I0 Z1 I2 I3', (-0.9355920202531878 + 0j)),
                               ('Y0 Y1 X2 X3', (0.3530829529141257 + 0j))),
                      'theta_sk': (FIND),
                      'factor': (0.0194148993856907 + 0j)}
                    ]
    }

    assert MANUAL_answer == X_sk_and_theta_sk