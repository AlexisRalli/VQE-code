from functools import reduce
from scipy.sparse import csr_matrix
from scipy.sparse import kron
import numpy as np
from scipy.sparse.linalg import expm, eigsh

from quchem.Misc_functions.Misc_functions import sparse_allclose

from openfermion.ops import QubitOperator
from openfermion.linalg import qubit_operator_sparse
from scipy.sparse import csc_matrix
from scipy.linalg import eigh


def Normalise_Clique(qubitOp_list):
    """
    Function takes in list of QubitOperators and returns a dictionary containing the normalised set of QubitOperators
    and correction term (gamma_l).

    Args:
        qubitOp_list (list): A list of QubitOperators

    Returns:
        dict: A dictionary of normalised terms (key = 'PauliWords') and correction factor (key = 'gamma_l')
    """
    factor = sum([np.abs(const)** 2 for qubitOp in qubitOp_list for PauliStrs, const in qubitOp.terms.items()])

    normalised_qubitOp_list = [QubitOperator(PauliStrs, const / np.sqrt(factor)) for qubitOp in qubitOp_list for
                               PauliStrs, const in qubitOp.terms.items()]

    return {'PauliWords': normalised_qubitOp_list, 'gamma_l': np.sqrt(factor)}

def Get_Xsk_op_list(anti_commuting_set, S_index, N_Qubits, check_reduction=False, atol=1e-8, rtol=1e-05):
    """
    Function to give all X_sk operators from a given anti_commuting set and S_index

    Args:
        anti_commuting_set(list): list of anti commuting QubitOperators
        S_index(int): index for Ps in anti_commuting_set list

    returns:
        X_sk_theta_sk_list(list): list of tuples containing X_sk QubitOperator and Theta_sk value
        normalised_FULL_set(dict): 'PauliWords' key gives NORMALISED terms that make up anti_commuting set
                                    'gamma_l' key gives normalization term
        Ps (QubitOperator): Pauli_S operator with cofactor of 1!
        gamma_l (float): normalization term

    """
    # 𝛾_𝑙 ∑ 𝛽_𝑗 𝑃_𝑗
    normalised_FULL_set = Normalise_Clique(anti_commuting_set)
    gamma_l = normalised_FULL_set['gamma_l']

    # ∑ 𝛽_𝑗 𝑃_𝑗
    norm_FULL_set = normalised_FULL_set['PauliWords'].copy()
    Pauli_S = norm_FULL_set.pop(S_index)  # removed from list!

    PauliStr_Ps, beta_S = tuple(*Pauli_S.terms.items())
    Ps = QubitOperator(PauliStr_Ps, 1) # new constant of 1

    X_sk_theta_sk_list = []
    for i, BetaK_Pk in enumerate(norm_FULL_set):
        Pk, BetaK = zip(*list(BetaK_Pk.terms.items()))

        X_sk = 1j * Ps * QubitOperator(Pk[0], 1) # new constant of 1

        if i < 1:
            theta_sk = np.arctan(BetaK[0] / beta_S)
            if beta_S.real < 0:
                # print('correcting quadrant')
                theta_sk = theta_sk + np.pi

            X_sk_theta_sk_list.append((X_sk, theta_sk))

            beta_S_new = np.sqrt(BetaK[0] ** 2 + beta_S ** 2)

            if not np.isclose((BetaK[0] * np.cos(theta_sk) - beta_S * np.sin(theta_sk)), 0):
                raise ValueError('mistake for choice of theta_sk')

        else:

            theta_sk = np.arctan(BetaK[0] / beta_S_new)
            X_sk_theta_sk_list.append((X_sk, theta_sk))

            if not np.isclose((BetaK[0] * np.cos(theta_sk) - beta_S_new * np.sin(theta_sk)), 0):
                raise ValueError('mistake for choice of theta_sk')

            beta_S_new = np.sqrt(BetaK[0] ** 2 + beta_S_new ** 2)


    ### check transformation - EXPENSIVE!
    if check_reduction:
        R_sk_list = []
        for X_sk_Op, theta_sk in X_sk_theta_sk_list:
            pauliword_X_sk_MATRIX = qubit_operator_sparse(QubitOperator(list(X_sk_Op.terms.keys())[0], -1j),
                                                          n_qubits=N_Qubits)
            const_X_sk = list(X_sk_Op.terms.values())[0]
            R_sk_list.append(expm(pauliword_X_sk_MATRIX * theta_sk / 2 * const_X_sk))

        R_S_matrix = reduce(np.dot, R_sk_list[::-1])  # <- note reverse order!

        Ps_mat = qubit_operator_sparse(Ps, n_qubits=N_Qubits)

        H_S = QubitOperator()
        for QubitOp in normalised_FULL_set['PauliWords']:
            H_S += QubitOp
        H_S_matrix = qubit_operator_sparse(H_S, n_qubits=N_Qubits)

        RHR = R_S_matrix.dot(H_S_matrix.dot(R_S_matrix.conj().transpose()))

        if not sparse_allclose(Ps_mat, RHR, atol=atol, rtol=rtol):
            raise ValueError('error in unitary partitioning reduction: R H_s R† != Ps')

    return X_sk_theta_sk_list, normalised_FULL_set, Ps, gamma_l


########## Linear Algebra approach


def Get_Rsl_matrix(Xsk_op_list, N_Qubits):

    """
    Function that gives matrix of Rsl from a list of X_sk operators, theta_sks. This is the output from Get_Xsk_op_list function.
    X_sk operators from a given anti_commuting set and S_index

    Args:
        X_sk_theta_sk_list(list): list of tuples containing X_sk QubitOperator and Theta_sk value
        N_Qubits (int): number of qubits

    returns:
        Rsl_matrix (np.sparse.csc_matrix)

    """

    R_sk_list = []
    for X_sk_Op, theta_sk in Xsk_op_list:
        pauliword_X_sk_MATRIX = qubit_operator_sparse(QubitOperator(list(X_sk_Op.terms.keys())[0], -1j),
                                                      n_qubits=N_Qubits)
        const_X_sk = list(X_sk_Op.terms.values())[0]
        
        R_sk_list.append(expm(pauliword_X_sk_MATRIX * theta_sk / 2 * const_X_sk))

    Rs_l_matrix = reduce(np.dot, R_sk_list[::-1])  # <- note reverse order!

    return Rs_l_matrix


def SeqRot_linalg_Energy(anti_commuting_sets, S_key_dict, N_Qubits, atol=1e-8, rtol=1e-05, check_reduction=False):
    # TODO: could return reduced_H_matrix sparse matrix!

    reduced_H_matrix = csc_matrix((2 ** N_Qubits, 2 ** N_Qubits), dtype=complex)

    H_single_terms = QubitOperator()

    for key in anti_commuting_sets:
        AC_set = anti_commuting_sets[key]

        if len(AC_set) < 2:
            H_single_terms += AC_set[0]
        else:
            S_index = S_key_dict[key]

            X_sk_theta_sk_list, full_normalised_set, Ps, gamma_l = Get_Xsk_op_list(AC_set, S_index, N_Qubits, check_reduction=check_reduction, atol=atol, rtol=rtol)


            R_S_matrix = Get_Rsl_matrix(X_sk_theta_sk_list, N_Qubits)
            Ps_mat = qubit_operator_sparse(Ps, n_qubits=N_Qubits)

            RPR_matrix = R_S_matrix.conj().transpose().dot(Ps_mat.dot(R_S_matrix))  # note this is R^{dag}PR and NOT: RHR^{dag}

            reduced_H_matrix += RPR_matrix * gamma_l

    reduced_H_matrix += qubit_operator_sparse(H_single_terms, n_qubits=N_Qubits)
    # eig_values, eig_vectors = sparse_eigs(reduced_H_matrix)
    if reduced_H_matrix.shape[0]<=64:
        eig_values, eig_vectors = eigh(reduced_H_matrix.todense()) # NOT sparse!
    else:
        eig_values, eig_vectors = eigsh(reduced_H_matrix, k=1, which='SA') # < solves eigenvalue problem for a complex Hermitian matrix.
    FCI_Energy = min(eig_values)
    return FCI_Energy