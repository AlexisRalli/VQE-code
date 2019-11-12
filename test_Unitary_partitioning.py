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
def test_Get_X_sk_operators_THETA_sk_values():
    """
    Standard use case. Making sure the correct theta_sk values are obtained!

    :return:
    """
    S = 0

    normalised_anti_commuting_sets ={
         0: {'PauliWords': [('I0 I1 Z2 Z3', (1 + 0j))],
             'factor': (1+0j)},

         1: {'PauliWords': [('Z0 I1 I2 I3', (0.8918294488900189 + 0j)),
                            ('Y0 X1 X2 Y3', (0.3198751585326103 + 0j)),
                            ('X0 I1 I2 I3', (0.3198751585326103 + 0j))],
             'factor': (0.9999999999999999+0j)},

         2: {'PauliWords': [('I0 Z1 I2 I3', (0.8283076631253103+0j)),
                            ('Y0 Y1 X2 X3', (-0.2970916080263448+0j)),
                            ('I0 X1 I2 I3', (-0.2970916080263448+0j)),
                            ('X0 Y1 I2 I3', (0.37064749842475486+0j))],
             'factor': (1.2913940071756902+0j)},
        }


    # note that normalised_anti_commuting_sets[1]['PauliWords'][0][1] is beta_s
    # look at eq. (16) ArXiv 1908.08067
    key1_tan_theta_01 = normalised_anti_commuting_sets[1]['PauliWords'][1][1]/normalised_anti_commuting_sets[1]['PauliWords'][0][1] #B_k/B_s
    key1_theta_01 = np.arctan(key1_tan_theta_01)
    beta_S_k1_NEW = np.sqrt(normalised_anti_commuting_sets[1]['PauliWords'][0][1]**2 + normalised_anti_commuting_sets[1]['PauliWords'][1][1]**2) # B_s^2 + B_k^2

    key1_tan_theta_02 = normalised_anti_commuting_sets[1]['PauliWords'][2][1]/beta_S_k1_NEW #B_k/B_s
    key1_theta_02 = np.arctan(key1_tan_theta_02)
    # beta_S_k2_NEW = np.sqrt(
    #     beta_S_k1_NEW + normalised_anti_commuting_sets[1]['PauliWords'][2][
    #         1] ** 2)
    # NOT needed... but included for reference!

    key2_tan_theta_01 =  normalised_anti_commuting_sets[2]['PauliWords'][1][1]/normalised_anti_commuting_sets[2]['PauliWords'][0][1] #B_k/B_s
    key2_theta_01 = np.arctan(key2_tan_theta_01)
    beta_S_k1_NEW = np.sqrt(
        normalised_anti_commuting_sets[2]['PauliWords'][0][1] ** 2 + normalised_anti_commuting_sets[2]['PauliWords'][1][
            1] ** 2)  # B_s^2 + B_k^2

    key2_tan_theta_02 = normalised_anti_commuting_sets[2]['PauliWords'][2][1]/beta_S_k1_NEW #B_k/B_s
    key2_theta_02 = np.arctan(key2_tan_theta_02)
    beta_S_k2_NEW = np.sqrt(
        beta_S_k1_NEW**2 + normalised_anti_commuting_sets[2]['PauliWords'][2][
            1] ** 2)

    key2_tan_theta_03 = normalised_anti_commuting_sets[2]['PauliWords'][3][1] / beta_S_k2_NEW  # B_k/B_s
    key2_theta_03 = np.arctan(key2_tan_theta_03)
    # beta_S_03_NEW = np.sqrt(
    #     beta_S_k2_NEW + normalised_anti_commuting_sets[2]['PauliWords'][3][
    #         1] ** 2)
    # NOT needed... but included for reference!

    MANUAL_answer = {1:
                         [
                             {'X_sk': (('Z0 I1 I2 I3', (0.8918294488900189+0j)),
                             ('Y0 X1 X2 Y3', (0.3198751585326103 + 0j))),
                             'theta_sk': (key1_theta_01),
                             'factor': (0.9999999999999999+0j)},

                             {'X_sk': (('Z0 I1 I2 I3', (0.8918294488900189+0j)),
                             ('X0 I1 I2 I3', (0.3198751585326103 + 0j))),
                             'theta_sk': (key1_theta_02),
                              'factor': (0.9999999999999999+0j)}
                         ],


                2:
                    [
                        {'X_sk': (('I0 Z1 I2 I3', (0.8283076631253103+0j)),
                               ('Y0 Y1 X2 X3', (-0.2970916080263448+0j))),
                      'theta_sk': (key2_theta_01),
                      'factor': (1.2913940071756902+0j)},

                        {'X_sk': (('I0 Z1 I2 I3', (0.8283076631253103+0j)),
                                  ('I0 X1 I2 I3', (-0.2970916080263448+0j))),
                         'theta_sk': (key2_theta_02),
                         'factor': (1.2913940071756902+0j)},

                        {'X_sk': (('I0 Z1 I2 I3', (0.8283076631253103 + 0j)),
                                  ('X0 Y1 I2 I3', (0.37064749842475486+0j))),
                         'theta_sk': (key2_theta_03),
                         'factor': (1.2913940071756902+0j)}

                    ]
    }

    X_sk_and_theta_sk = Get_X_sk_operators(normalised_anti_commuting_sets, S=S)



    assert X_sk_and_theta_sk == MANUAL_answer


##
def test_convert_X_sk():
    X_sk = (
              ('I0 I1 I2 I3 X4 X5 X6 X7 Y8 Y9 Y10 Y11 Z12 Z13 Z14 Z15', (0.8918294488900189+0j)),
              ('I0 X1 Y2 Z3 I4 X5 Y6 Z7 I8 X9 Y10 Z11 I12 X13 Y14 Z15', (0.3198751585326103+0j))
            )

    New_PauliWord = convert_X_sk(X_sk)
    # (cofactor, New_Pauli_string)  #QubitNo    #new constant

    cofactor = X_sk[0][1] * X_sk[1][1]
    Correct_New_Pauli = [
        ((1, 'I'), '0', cofactor),
        ((1, 'X'), '1', cofactor),
        ((1, 'Y'), '2', cofactor),
        ((1, 'Z'), '3', cofactor),
        ((1, 'X'), '4', cofactor),
        ((1, 'I'), '5', cofactor),
        ((1j, 'Z'), '6', cofactor),
        (((-0 - 1j), 'Y'), '7', cofactor),
        ((1, 'Y'), '8', cofactor),
        (((-0 - 1j), 'Z'), '9', cofactor),
        ((1, 'I'), '10', cofactor),
        ((1j, 'X'), '11', cofactor),
        ((1, 'Z'), '12', cofactor),
        ((1j, 'Y'), '13', cofactor),
        (((-0 - 1j), 'X'), '14', cofactor),
        ((1, 'I'), '15', cofactor)
                    ]
    assert New_PauliWord == Correct_New_Pauli


# pp = Get_beta_j_cofactors({
#     0: [('I0 I1 Z2 Z3', (1 + 0j))],
#     1: [('Z0 I1 I2 I3', (0.8918294488900189 + 0j)),
#         ('Y0 X1 X2 Y3', (0.3198751585326103 + 0j)),
#         ('X0 I1 I2 I3', (0.3198751585326103 + 0j))],
#
#     2: [('I0 Z1 I2 I3', (0.9412848366792171 + 0j)),
#         ('Y0 Y1 X2 X3', (-0.33761347164735517 + 0j)),
#         ('I0 X1 I2 I3', (-0.33761347164735517 + 0j)),
#         ('X0 Y1 I2 I3', (0.421202031023012301 + 0j))]
# })