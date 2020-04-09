from openfermion.ops import QubitOperator
import numpy as np


def Multiply_PauliQubitOps(qubitOp_1, qubitOp_2, mulitplying_const=1):
    """

    TODO

    NOTE this function does NOT!!! repeat not multiply by the qubitOp_2 constnat!

    Args:
        qubitOp_1 ():
        qubitOp_2 ():

    Returns:
        tuple:


    """
    convert_term = {
        'II': (1, 'I'),
        'IX': (1, 'X'),
        'IY': (1, 'Y'),
        'IZ': (1, 'Z'),

        'XI': (1, 'X'),
        'XX': (1, 'I'),
        'XY': (1j, 'Z'),
        'XZ': (-1j, 'Y'),

        'YI': (1, 'Y'),
        'YX': (-1j, 'Z'),
        'YY': (1, 'I'),
        'YZ': (1j, 'X'),

        'ZI': (1, 'Z'),
        'ZX': (1j, 'Y'),
        'ZY': (-1j, 'X'),
        'ZZ': (1, 'I')
    }

    PauliStr_1_tuples_P1 = [tup for PauliStrs, const in qubitOp_1.terms.items() for tup in PauliStrs]
    qubitNo_P1, PauliStr_P1 = zip(*PauliStr_1_tuples_P1)
    qubitNo_P1 = np.array(qubitNo_P1)
    qubitNo_P1_CONST = list(qubitOp_1.terms.values())[0]

    PauliStr_tuples_P2 = [tup for PauliStrs, const in qubitOp_2.terms.items() for tup in PauliStrs]
    qubitNo_P2, PauliStr_P2 = zip(*PauliStr_tuples_P2)
    qubitNo_P2 = np.array(qubitNo_P2)
    qubitNo_P2_CONST = list(qubitOp_2.terms.values())[0]

    common_qubits = np.intersect1d(qubitNo_P1, qubitNo_P2)

    PauliStr_P1_common = np.take(PauliStr_P1, np.where(np.isin(qubitNo_P1, common_qubits) == True)).flatten()
    PauliStr_P2_common = np.take(PauliStr_P2, np.where(np.isin(qubitNo_P2, common_qubits) == True)).flatten()

    new_paulistr_list = []
    new_factor = []
    for index, pauli_str_P1 in enumerate(PauliStr_P1_common):

        pauli_str_P2 = PauliStr_P2_common[index]
        qubitNo = common_qubits[index]

        combined_pauli_str = pauli_str_P1 + pauli_str_P2

        if convert_term[combined_pauli_str][1] != 'I':
            new_pauli_str = convert_term[combined_pauli_str][1] + str(qubitNo)
            new_paulistr_list.append(new_pauli_str)

            new_factor.append(convert_term[combined_pauli_str][0])

    new_constant = np.prod(new_factor) * qubitNo_P1_CONST * mulitplying_const  # * qubitNo_P2_CONST

    for index, qubitNo in enumerate(qubitNo_P1):
        if qubitNo not in common_qubits:
            Paulistring_P1 = PauliStr_P1[index]
            new_paulistr_list.append(Paulistring_P1 + str(qubitNo))

    for index, qubitNo in enumerate(qubitNo_P2):
        if qubitNo not in common_qubits:
            Paulistring_P2 = PauliStr_P2[index]
            new_paulistr_list.append(Paulistring_P2 + str(qubitNo))

    seperator = ' '
    pauliStr_list = seperator.join(new_paulistr_list)

    New_P = QubitOperator(pauliStr_list, new_constant)

    return New_P


from quchem.Unitary_partitioning import *


def Get_X_SET(anti_commuting_set, N_index):
    """
    X = i ( ∑_{k=1}^{n-1} B_{k} P_{k} ) P_{n}

    X =  i( ∑_{k=1}^{n-1} B_{k} P_{kn}

        where P_{ks} = P_{k} * P_{n}

    note ∑_{k=1}^{n-1} B_{k}^{2} = 1

    therefore have:
    X =  gamma_l * i( ∑_{k=1}^{n-1} B_{k} P_{kn}


    Args:
        anti_commuting_set (list):
        S_index (int):
        no_qubits (int):
    Returns:
        LCU_dict (dict): A dictionary containing the linear combination of terms required to perform R ('R_LCU')
                         the correction fsinactors to make all real and positive ('LCU_correction')
                         the angle to perform R gate ('alpha')
                         the PauliS term ('P_s')
     """

    # 𝛾_𝑙 ∑ 𝛽_𝑗 𝑃_𝑗
    normalised_FULL_set = Get_beta_j_cofactors(anti_commuting_set)
    gamma_l = normalised_FULL_set['gamma_l']

    norm_FULL_set = normalised_FULL_set['PauliWords'].copy()

    # 𝛽_n 𝑃_n
    qubitOp_Pn_beta_n = norm_FULL_set.pop(N_index)

    # Ω_𝑙 ∑ 𝛿_𝑗 𝑃_𝑗  ... note this doesn't contain 𝛽_n 𝑃_n
    H_n_1 = Get_beta_j_cofactors(norm_FULL_set)
    Omega_l = H_n_1['gamma_l']

    # cos(𝜙_{𝑛−1}) =𝛽_𝑛
    phi_n_1 = np.arccos(list(qubitOp_Pn_beta_n.terms.values())[0])
    #     phi_n_1 = np.arcsin(Omega_l)

    # 𝑖 ∑ 𝛿_{𝑘} 𝑃_{𝑘𝑛}
    X_set = {}
    X_set['X_PauliWords'] = []
    for qubitOp_Pk in H_n_1['PauliWords']:
        new_PauliWord = Multiply_PauliQubitOps(qubitOp_Pk, qubitOp_Pn_beta_n,
                                               mulitplying_const=1j)  # here we times by 1j due to defintion of X
        X_set['X_PauliWords'].append(new_PauliWord)

    if not np.isclose(sum(np.absolute(list(qubitOp.terms.values())[0]) ** 2 for qubitOp in X_set['X_PauliWords']), 1):
        raise ValueError('normalisation of X operator incorrect: {}'.format(
            sum(list(qubitOp.terms.values())[0] ** 2 for qubitOp in X_set['X_PauliWords'])))

    # THIS IS NOT NEED BUT I AM USING TO CHECK
    X_set['H_n'] = [QubitOperator(qubitOp, const * np.sin(phi_n_1))
                    for operator in H_n_1['PauliWords'] for qubitOp, const in operator.terms.items()] + [
                       QubitOperator(list(qubitOp_Pn_beta_n.terms.keys())[0], np.cos(phi_n_1))]

    if not np.isclose(sum(np.absolute(list(qubitOp.terms.values())[0]) ** 2 for qubitOp in X_set['H_n']), 1):
        raise ValueError('normalisation of H_n operator incorrect: {}'.format(
            sum(np.absolute(list(qubitOp.terms.values())[0]) ** 2 for qubitOp in X_set['H_n'])))
    # THIS IS NOT NEED BUT I AM USING TO CHECK

    if not np.isclose((list(qubitOp_Pn_beta_n.terms.values())[0] ** 2 + Omega_l ** 2), 1):
        raise ValueError('Ω^2 + 𝛽n^2 does NOT equal 1')

    #     if list(qubitOp_Pn_beta_n.terms.values())[0]<0:
    #         X_set['P_n'] = QubitOperator(list(qubitOp_Pn_beta_n.terms.keys())[0], -1)
    #     else:
    #         X_set['P_n'] = QubitOperator(list(qubitOp_Pn_beta_n.terms.keys())[0], 1)

    X_set['P_n'] = QubitOperator(list(qubitOp_Pn_beta_n.terms.keys())[0], 1)

    #     if list(qubitOp_Pn_beta_n.terms.values())[0]<0:
    #         X_set['gamma_l'] = gamma_l *-1
    #     else:
    #          X_set['gamma_l'] = gamma_l

    X_set['gamma_l'] = gamma_l
    X_set['H_n_1'] = H_n_1['PauliWords']
    X_set['Omega_l'] = Omega_l
    X_set['phi_n_1'] = phi_n_1
    return X_set