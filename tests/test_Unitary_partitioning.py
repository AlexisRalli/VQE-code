from quchem.Unitary_partitioning import *

import cirq
import pytest
import numpy as np
# in Terminal run (LINUX!):
# py.test /home/alexisr/Documents/PhD/Code/PhD\ Code/test_Unitary_partitioning.py -v

###
def test_Get_beta_j_cofactors_normalisation():
    '''
    Standard use case.

    Here checking if  (sum_j B_j^2 = 1) according to eq (10) in ArXiv:1908.08067

    :return:
    '''

    anti_commuting_set =  [('Z0 I1 I2 I3', (0.1371657293179602+0j)), ('Y0 X1 X2 Y3', (0.04919764587885283+0j)), ('X0 I1 I2 I3', (0.04919764587885283+0j))]

    normalised_anticommuting_set_DICT = Get_beta_j_cofactors(anti_commuting_set)

    sum_squares = sum([constant ** 2 for PauliWord, constant in normalised_anticommuting_set_DICT['PauliWords']])
    # if not np.isclose(1 + 0j, sum_squares, rtol=1e-09, atol=0.0):
    #     raise ValueError('B_j^2 terms sum too {} rather than 1'.format(sum_squares))
    assert np.isclose(1 + 0j, sum_squares, rtol=1e-09, atol=0.0)

def test_Get_beta_j_cofactors_GAMMA_L():
    '''
    Standard use case.

    Makes sure that factor can be used to re-gain old constant!
    [old constant = [gamma_l * new_constant]

    :return:
    '''
    anti_commuting_set =  [('Z0 I1 I2 I3', (0.1371657293179602+0j)), ('Y0 X1 X2 Y3', (0.04919764587885283+0j)), ('X0 I1 I2 I3', (0.04919764587885283+0j))]
    normalised_anticommuting_set_DICT = Get_beta_j_cofactors(anti_commuting_set)

    gamma_l = normalised_anticommuting_set_DICT['gamma_l']
    regaining_old_term=[]
    for PauliWord, constant in normalised_anticommuting_set_DICT['PauliWords']:
        regaining_old_term.append((PauliWord, constant*gamma_l))

    assert regaining_old_term == anti_commuting_set

def test_Get_beta_j_cofactors_single_term():
    '''
    test for when one thing in anti_commuting_set.

    :return:
    '''
    anti_commuting_set =  [('I0 I1 I2 I3', (-0.32760818995565577+0j))]
    normalised_anticommuting_set_DICT = Get_beta_j_cofactors(anti_commuting_set)

    factor = anti_commuting_set[0][1]**2
    new_constant = anti_commuting_set[0][1] / np.sqrt(factor)
    expected = {'PauliWords': [(anti_commuting_set[0][0], new_constant)], 'gamma_l': np.sqrt(factor)}

    assert normalised_anticommuting_set_DICT == expected


###
def test_Get_X_sk_operators_THETA_sk_values():
    """
    Standard use case. Making sure the correct theta_sk values are obtained!

    :return:
    """
    S = 0

    normalised_anticommuting_set_DICT ={
                                        'PauliWords': [   ('Z0 I1 I2 I3', (0.8918294488900189+0j)),
                                                          ('Y0 X1 X2 Y3', (0.3198751585326103+0j)),
                                                          ('X0 I1 I2 I3', (0.3198751585326103+0j))   ],
                                        'gamma_l': (0.1538026463340925+0j)
                                    }

    X_sk_theta_sk = Get_X_sk_operators(normalised_anticommuting_set_DICT, S=0)
#

    # note that normalised_anti_commuting_sets[1]['PauliWords'][0][1] is beta_s
    # look at eq. (16) ArXiv 1908.08067
    PauliWord_S = normalised_anticommuting_set_DICT['PauliWords'][S]
    beta_S = PauliWord_S[1]

    beta_kequal1 = normalised_anticommuting_set_DICT['PauliWords'][1][1]
    tan_theta_S1 = beta_kequal1 / beta_S
    theta_01 = np.arctan(tan_theta_S1)
    beta_S = beta_kequal1*np.sin(theta_01) + beta_S*np.cos(theta_01)


    beta_kequal2 = normalised_anticommuting_set_DICT['PauliWords'][2][1]
    tan_theta_S2 = beta_kequal2 / beta_S
    theta_02 = np.arctan(tan_theta_S2)
    beta_S = beta_kequal1*np.sin(theta_02) + beta_S*np.cos(theta_02)

    expected = {'X_sk_theta_sk': [{'X_sk': convert_X_sk((PauliWord_S, normalised_anticommuting_set_DICT['PauliWords'][1])),
                                   'theta_sk': theta_01},
                                 {'X_sk': convert_X_sk((PauliWord_S, normalised_anticommuting_set_DICT['PauliWords'][2])),
                                    'theta_sk': theta_02}],

                'PauliWord_S': (PauliWord_S[0], beta_S),
                'gamma_l': normalised_anticommuting_set_DICT['gamma_l']}

    assert X_sk_theta_sk == expected

###
def test_convert_X_sk_normal():
    """
    Standard use case

    :return:
    """
    # contains all Pauli relations:
    X_sk = (
              ('I0 I1 I2 I3 X4 X5 X6 X7 Y8 Y9 Y10 Y11 Z12 Z13 Z14 Z15', (0.8918294488900189+0j)),
              ('I0 X1 Y2 Z3 I4 X5 Y6 Z7 I8 X9 Y10 Z11 I12 X13 Y14 Z15', (0.3198751585326103+0j))
            )

    convert_term ={
        'II': (1,'I'),
        'IX': (1,'X'),
        'IY': (1,'Y'),
        'IZ': (1,'Z'),

        'XI': (1,'X'),
        'XX': (1,'I'),
        'XY': (1j,'Z'),
        'XZ': (-1j,'Y'),

        'YI': (1,'Y'),
        'YX': (-1j,'Z'),
        'YY': (1,'I'),
        'YZ': (1j,'X'),

        'ZI': (1,'Z'),
        'ZX': (1j,'Y'),
        'ZY': (-1j,'X'),
        'ZZ': (1,'I')
    }
    cofactor_SIGN = np.prod([Pauli_factor_string[0] for PauliCombo, Pauli_factor_string in convert_term.items()])
    cofactor = 1j

    Correct_New_Pauli = ('I0 X1 Y2 Z3 X4 I5 Z6 Y7 Y8 Z9 I10 X11 Z12 Y13 X14 I15', cofactor * cofactor_SIGN)

    New_PauliWord = convert_X_sk(X_sk)
    assert New_PauliWord == Correct_New_Pauli

def test_convert_X_sk_non_pauli():
    """
    Checking if non Pauli Operators used a KeyError is given!

    :return:
    """
    X_sk = (
              ('Z0 W1 I2 I3', (0.8918294488900189+0j)),
              ('Y0 X1 X2 Y3', (0.3198751585326103+0j))
            )

    with pytest.raises(KeyError) as excinfo:
        assert convert_X_sk(X_sk) in excinfo.value

###
def test_My_R_sk_Gate_NON_DAGGER():
    theta_sk = np.pi
    R_S = My_R_sk_Gate(theta_sk, dagger=False)
    R_S_matrix = R_S._unitary_()

    circuit_NON_DAGGER = cirq.Rz(theta_sk)**-1
    matrix_NON_DAGGER = circuit_NON_DAGGER._unitary_()

    assert np.isclose(matrix_NON_DAGGER, R_S_matrix).all()

def test_My_R_sk_Gate_DAGGER():
    theta_sk = np.pi
    R_S_DAGGER = My_R_sk_Gate(theta_sk, dagger=True)
    R_S_DAGGER_matrix = R_S_DAGGER._unitary_()

    circuit_DAGGER = cirq.Rz(theta_sk)
    matrix_DAGGER = circuit_DAGGER._unitary_()

    assert np.isclose(matrix_DAGGER, R_S_DAGGER_matrix).all()

###

def test_Get_R_S_operators():
    """
    # currently WRONG
    This is due to my R_sk gate having a different HASH. Actually is correct!
    #check outputs!

    :return:
    """

    X_sk_and_theta_sk = {'X_sk_theta_sk': [{'X_sk': ('X0 X1 X2 Y3', (0.28527408634774526 + 0j)),
                            'theta_sk': (0.34438034648829496 + 0j)},
                           {'X_sk': ('Y0 I1 I2 I3', (-0.28527408634774526 + 0j)),
                            'theta_sk': (0.3423076794345934 + 0j)}],
         'PauliWord_S': ('Z0 I1 I2 I3', (1 + 0j)),
         'gamma_l': (0.1538026463340925 + 0j)}

    dagger = False

    R_S_operators = Get_R_S_operators(X_sk_and_theta_sk, dagger=dagger)

    for i in range(len(R_S_operators)):
        circ_gen = R_S_operators[i]['q_circuit']
        circuit = cirq.Circuit.from_ops(cirq.decompose_once(
            (circ_gen(*cirq.LineQubit.range(circ_gen.num_qubits())))))
        circuit_oper = list(circuit.all_operations())

        R_S_operators[i]['q_circuit'] = circuit_oper



    list_quantum_circuits_and_gammal = []
    for terms in X_sk_and_theta_sk['X_sk_theta_sk']:
        R_s_k_circuit_instance = R_sk_full_circuit(terms['X_sk'], terms['theta_sk'], dagger=dagger)

        circuit = cirq.Circuit.from_ops(cirq.decompose_once(
            (R_s_k_circuit_instance(*cirq.LineQubit.range(R_s_k_circuit_instance.num_qubits())))))
        circuit_oper = list(circuit.all_operations())

        correction_factor = X_sk_and_theta_sk['gamma_l']
        list_quantum_circuits_and_gammal.append({'q_circuit': circuit_oper, 'gamma_l': correction_factor})

    assert R_S_operators == list_quantum_circuits_and_gammal

###

def test_unitary_partitioning_method():
    anti_commuting_set = [('I0 I1 I2 Z3', (-0.2234315367466397+0j)),
                            ('X0 Y1 Y2 X3', (0.04530261550868928+0j))]

    normalised_set = Get_beta_j_cofactors(anti_commuting_set)

    P_s = normalised_set['PauliWords'][0]
    P_k = normalised_set['PauliWords'][1]

    gamma_l = normalised_set['gamma_l']
    i_Xsk = convert_X_sk((P_s, P_k))
    theta_sk = np.arctan(P_k[1]/P_s[1])

    # new beta_s
    beta_s = P_k[1]*np.sin(theta_sk) + P_s[1]*np.cos(theta_sk)

    # # WRONG method to get new beta_s
    # P_s[1] = np.sqrt((P_s[1]** 2 + P_k[1]**2))

    check = Get_X_sk_operators(normalised_set, S=0)

    assert check['PauliWord_S'] == (P_s[0], beta_s) and check['X_sk_theta_sk'][0]['theta_sk'] == theta_sk and check['gamma_l'] == gamma_l and check['X_sk_theta_sk'][0]['X_sk'] == i_Xsk


# def test_R_sk_full_circuit():
#
#     anti_commuting_set = [('I0 I1 I2 Z3', (-0.2234315367466397+0j)),
#                             ('X0 Y1 Y2 X3', (0.04530261550868928+0j))]
#
#     normalised_set = Get_beta_j_cofactors(anti_commuting_set)
#
#     X_sk_Ops = Get_X_sk_operators(normalised_set, S=0)
#
#     X_SK_Test = X_sk_Ops['X_sk_theta_sk'][0]['X_sk']
#     theta_sk = X_sk_Ops['X_sk_theta_sk'][0]['theta_sk']
#
#     R_sk_full_DAGGER = R_sk_full_circuit(X_SK_Test, theta_sk, dagger=True)
#     # print(cirq.Circuit.from_ops(cirq.decompose_once((R_sk_full(*cirq.LineQubit.range(R_sk_full.num_qubits()))))))

def test_unitary_partitioning_method_VS_STANDARD():
    from quchem.quantum_circuit_functions import *
    import cirq
    num_shots=10000

    anti_commuting_set = [('I0 I1 I2 Z3', (-0.2234315367466397+0j)),
                            ('X0 Y1 Y2 X3', (0.04530261550868928+0j))]

    normalised_set = Get_beta_j_cofactors(anti_commuting_set)

    X_sk_Ops = Get_X_sk_operators(normalised_set, S=0)

    R_sk_full_DAGGER = Get_R_S_operators(X_sk_Ops, dagger=True)
    R_sk_full = Get_R_S_operators(X_sk_Ops, dagger=False)

    ANSATZ = [cirq.X.on(cirq.LineQubit(2)),
                           cirq.X.on(cirq.LineQubit(3)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(0)),
                           cirq.H.on(cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.Rz(np.pi * 0.0).on(cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(0)),
                           cirq.H.on(cirq.LineQubit(2)),
                           cirq.H.on(cirq.LineQubit(0)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.Rz(np.pi * 0.0).on(cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.H.on(cirq.LineQubit(0)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(2)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(1)),
                           cirq.H.on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.Rz(np.pi * 0.3183098861837907).on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(1)),
                           cirq.H.on(cirq.LineQubit(3)),
                           cirq.H.on(cirq.LineQubit(1)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.Rz(np.pi * -0.3183098861837907).on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.H.on(cirq.LineQubit(1)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(3)),
                           cirq.H.on(cirq.LineQubit(0)),
                           cirq.H.on(cirq.LineQubit(1)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(2)),
                           cirq.H.on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.Rz(np.pi * 0.6366197723675814).on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.H.on(cirq.LineQubit(0)),
                           cirq.H.on(cirq.LineQubit(1)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(2)),
                           cirq.H.on(cirq.LineQubit(3)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(0)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(1)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(2)),
                           cirq.H.on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.Rz(np.pi * -0.6366197723675814).on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(0)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(1)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(2)),
                           cirq.H.on(cirq.LineQubit(3)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(0)),
                           cirq.H.on(cirq.LineQubit(1)),
                           cirq.H.on(cirq.LineQubit(2)),
                           cirq.H.on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.Rz(np.pi * -0.6366197723675814).on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(0)),
                           cirq.H.on(cirq.LineQubit(1)),
                           cirq.H.on(cirq.LineQubit(2)),
                           cirq.H.on(cirq.LineQubit(3)),
                           cirq.H.on(cirq.LineQubit(0)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(1)),
                           cirq.H.on(cirq.LineQubit(2)),
                           cirq.H.on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.Rz(np.pi * -0.6366197723675814).on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.H.on(cirq.LineQubit(0)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(1)),
                           cirq.H.on(cirq.LineQubit(2)),
                           cirq.H.on(cirq.LineQubit(3)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(0)),
                           cirq.H.on(cirq.LineQubit(1)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(2)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.Rz(np.pi * 0.6366197723675814).on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(0)),
                           cirq.H.on(cirq.LineQubit(1)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(2)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(3)),
                           cirq.H.on(cirq.LineQubit(0)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(1)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(2)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.Rz(np.pi * 0.6366197723675814).on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.H.on(cirq.LineQubit(0)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(1)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(2)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(3)),
                           cirq.H.on(cirq.LineQubit(0)),
                           cirq.H.on(cirq.LineQubit(1)),
                           cirq.H.on(cirq.LineQubit(2)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.Rz(np.pi * 0.6366197723675814).on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.H.on(cirq.LineQubit(0)),
                           cirq.H.on(cirq.LineQubit(1)),
                           cirq.H.on(cirq.LineQubit(2)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(3)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(0)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(1)),
                           cirq.H.on(cirq.LineQubit(2)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.Rz(np.pi * -0.6366197723675814).on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(0)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(1)),
                           cirq.H.on(cirq.LineQubit(2)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(3))]
    full_anstaz_circuit = cirq.Circuit.from_ops(*ANSATZ)


    term_reduction_circuits_first = [cirq.decompose_once(
    (term['q_circuit'](*cirq.LineQubit.range(term['q_circuit'].num_qubits())))) for term in R_sk_full_DAGGER]

    Pauliword_S = X_sk_Ops['PauliWord_S']
    q_circuit_Pauliword_S_object = Perform_PauliWord(Pauliword_S)

    q_circuit_Pauliword_S = cirq.Circuit.from_ops(
        cirq.decompose_once((q_circuit_Pauliword_S_object(
            *cirq.LineQubit.range(q_circuit_Pauliword_S_object.num_qubits())))))


    term_reduction_circuits_LAST = [cirq.decompose_once(
        (term['q_circuit'](*cirq.LineQubit.range(term['q_circuit'].num_qubits())))) for term in R_sk_full]

    q_circuit_change_basis_and_measure = Change_Basis_and_Measure_PauliWord(Pauliword_S)

    q_circuit_Pauliword_S_change_basis_and_measure = cirq.Circuit.from_ops(
        cirq.decompose_once(
            (q_circuit_change_basis_and_measure(
                *cirq.LineQubit.range(q_circuit_change_basis_and_measure.num_qubits())))))

    full_circuit = cirq.Circuit.from_ops(
        [
            *full_anstaz_circuit.all_operations(),
            *term_reduction_circuits_first,
            *q_circuit_Pauliword_S.all_operations(),
            *term_reduction_circuits_LAST,
            *q_circuit_Pauliword_S_change_basis_and_measure.all_operations()
        ]
    )

    unintary_part_dict = {0: {'circuit': full_circuit, 'PauliWord': Pauliword_S[0],
                              'gamma_l': normalised_set['gamma_l']*Pauliword_S[1]}
                          }
    UP = Simulation_Quantum_Circuit_Dict(unintary_part_dict, num_shots)
    print(UP.Calc_energy_via_parity())

    from quchem.standard_method import *
    standard_method = Get_quantum_circuits_and_constants_NORMAL(full_anstaz_circuit, anti_commuting_set)

    SM = Simulation_Quantum_Circuit_Dict(standard_method, num_shots)
    print(SM.Calc_energy_via_parity())

    # print(
    #     cirq.Circuit.from_ops(
    #         cirq.decompose_once((R_sk_full_DAGGER[0]['q_circuit'](
    #             *cirq.LineQubit.range(R_sk_full_DAGGER[0]['q_circuit'].num_qubits()))))))

