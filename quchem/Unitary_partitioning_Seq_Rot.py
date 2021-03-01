from functools import reduce
from scipy.sparse import csr_matrix
from scipy.sparse import kron
import numpy as np


from openfermion.ops import QubitOperator


def Get_pauli_matrix(PauliOp, N_system_qubits):
    pauliDict = {'X': np.array([[0, 1], [1, 0]]),
                 'Y': np.array([[0, -1j], [1j, 0]]),
                 'Z': np.array([[1, 0], [0, -1]]),
                 'I': np.eye(2)}

    list_Q_nos, list_P_strs = list(zip(*[Paulistrs for Paulistrs, const in PauliOp.terms.items()][0]))

    list_of_ops = []
    for i in range(N_system_qubits):
        if i in list_Q_nos:
            index = list_Q_nos.index(i)
            list_of_ops.append(pauliDict[list_P_strs[index]])
        else:
            list_of_ops.append(pauliDict['I'])

    matrix = reduce(kron, list_of_ops)

    return matrix

def Get_beta_j_cofactors(qubitOp_list):
    """
    Function takes in list of QubitOperators and returns a dictionary containing the normalised set of QubitOperators
    and correction term (gamma_l).

    Args:
        qubitOp_list (list): A list of QubitOperators

    Returns:
        dict: A dictionary of normalised terms (key = 'PauliWords') and correction factor (key = 'gamma_l')
    """
    factor = sum([const ** 2 for qubitOp in qubitOp_list for PauliStrs, const in qubitOp.terms.items()])

    normalised_qubitOp_list = [QubitOperator(PauliStrs, const / np.sqrt(factor)) for qubitOp in qubitOp_list for
                               PauliStrs, const in qubitOp.terms.items()]

    return {'PauliWords': normalised_qubitOp_list, 'gamma_l': np.sqrt(factor)}

def Get_Xsk_op_list(anti_commuting_set, S_index):
    """
    Function to give all X_sk operators from a given anti_commuting set and S_index

    Args:
        anti_commuting_set(list): list of anti commuting QubitOperators
        S_index(int): index for Ps in anti_commuting_set list

    returns:
        X_sk_theta_sk(list): list of tuples containing X_sk QubitOperator and Theta_sk value
        normalised_FULL_set(dict): 'PauliWords' key gives NORMALISED terms that make up anti_commuting set
                                    'gamma_l' key gives normalization term
        Ps (QubitOperator): Pauli_S operator with cofactor of 1!
        gamma_l (float): normalization term

    """
    # 𝛾_𝑙 ∑ 𝛽_𝑗 𝑃_𝑗
    normalised_FULL_set = Get_beta_j_cofactors(anti_commuting_set)
    gamma_l = normalised_FULL_set['gamma_l']

    # ∑ 𝛽_𝑗 𝑃_𝑗
    norm_FULL_set = normalised_FULL_set['PauliWords'].copy()
    Pauli_S = norm_FULL_set.pop(S_index)  # removed from list!

    Ps = QubitOperator(list(Pauli_S.terms.keys())[0], 1)
    beta_S = list(Pauli_S.terms.values())[0]

    X_sk_theta_sk = []
    for i, BetaK_Pk in enumerate(norm_FULL_set):
        Pk, BetaK = zip(*list(BetaK_Pk.terms.items()))

        X_sk = 1j * Ps * QubitOperator(Pk[0], 1)

        if i < 1:
            theta_sk = np.arctan(BetaK[0] / beta_S)
            if beta_S.real < 0:
                # print('correcting quadrant')
                theta_sk = theta_sk + np.pi

            X_sk_theta_sk.append((X_sk, theta_sk))

            beta_S_new = np.sqrt(BetaK[0] ** 2 + beta_S ** 2)

            if not np.isclose((BetaK[0] * np.cos(theta_sk) - beta_S * np.sin(theta_sk)), 0):
                raise ValueError('mistake for choice of theta_sk')

        else:

            theta_sk = np.arctan(BetaK[0] / beta_S_new)
            X_sk_theta_sk.append((X_sk, theta_sk))

            if not np.isclose((BetaK[0] * np.cos(theta_sk) - beta_S_new * np.sin(theta_sk)), 0):
                raise ValueError('mistake for choice of theta_sk')

            beta_S_new = np.sqrt(BetaK[0] ** 2 + beta_S_new ** 2)

    return X_sk_theta_sk, normalised_FULL_set, Ps, gamma_l

from quchem.quantum_circuit_functions import *
def Build_reduction_circuit(anti_commuting_set, S_index, check_reduction=False):
    """
    Function to build R_S (make up of all R_SK terms)

    Args:
        anti_commuting_set(list): list of anti commuting QubitOperators
        S_index(int): index for Ps in anti_commuting_set list
        check_reduction (optional, bool): use linear algebra to check that 𝑅s† 𝐻s 𝑅s == 𝑃s
    returns:
        full_RS_circuit(cirq.Circuit): Q_circuit for R_s operator
        Ps (QubitOperator): Pauli_S operator with cofactor of 1!
        gamma_l (float): normalization term

    """
    X_sk_theta_sk_list, full_normalised_set, Ps, gamma_l = Get_Xsk_op_list(anti_commuting_set, S_index)

    circuit_list = []
    for X_sk_Op, theta_sk in X_sk_theta_sk_list:
        pauliword_X_sk = list(X_sk_Op.terms.keys())[0]
        const_X_sk = list(X_sk_Op.terms.values())[0]

        full_exp_circ_obj = full_exponentiated_PauliWord_circuit(QubitOperator(pauliword_X_sk, -1j),
                                                                 theta_sk / 2 * const_X_sk)

        circuit = cirq.Circuit(
            cirq.decompose_once((full_exp_circ_obj(*cirq.LineQubit.range(full_exp_circ_obj.num_qubits())))))

        circuit_list.append(circuit)

    full_RS_circuit = cirq.Circuit(circuit_list)

    if check_reduction:

        H_S = QubitOperator()
        for QubitOp in full_normalised_set['PauliWords']:
            H_S += QubitOp
        from openfermion import qubit_operator_sparse
        H_S_matrix = qubit_operator_sparse(H_S)

        n_qubits =  int(np.log2(qubit_operator_sparse(H_S).todense().shape[0]))
        qbits = cirq.LineQubit.range(n_qubits)

        R_S_matrix = full_RS_circuit.unitary(qubits_that_should_be_present=qbits)

        # Ps_mat = Get_pauli_matrix(Ps, len(qbits))

        Ps_mat=qubit_operator_sparse(Ps, n_qubits=len(qbits))
        reduction_mat = R_S_matrix.dot(H_S_matrix.dot(R_S_matrix.conj().transpose()))

        if not (np.allclose(Ps_mat.todense(), reduction_mat)):
            print('reduction circuit incorrect...   𝑅s 𝐻s 𝑅s† != 𝑃s')

    return full_RS_circuit, Ps, gamma_l


def Generate_Full_Q_Circuit_conj(Full_Ansatz_Q_Circuit, anti_commuting_set, S_index, check_reduction=False):
    """
    Function to build full Q Circuit... ansatz circuit + R_S

    Args:
        Full_Ansatz_Q_Circuit (cirq.Circuit): ansatz quantum circuit
        anti_commuting_set(list): list of anti commuting QubitOperators
        S_index(int): index for Ps in anti_commuting_set list
        check_reduction (optional, bool): use linear algebra to check that 𝑅s† 𝐻s 𝑅s == 𝑃s
    returns:
        full_RS_circuit(cirq.Circuit): Q_circuit for R_s operator
        Ps (QubitOperator): Pauli_S operator with cofactor of 1!
        gamma_l (float): normalization term

    """
    Reduction_circuit_circ, Ps, gamma_l = Build_reduction_circuit(anti_commuting_set, S_index,
                                                                  check_reduction=check_reduction)

    measure_PauliS_in_Z_basis_obj = change_pauliword_to_Z_basis_then_measure(Ps)
    measure_PauliS_in_Z_basis_Q_circ = cirq.Circuit(cirq.decompose_once(
        (measure_PauliS_in_Z_basis_obj(*cirq.LineQubit.range(measure_PauliS_in_Z_basis_obj.num_qubits())))))

    full_circuit = cirq.Circuit(
        [
            Full_Ansatz_Q_Circuit.all_operations(),
            *Reduction_circuit_circ.all_operations(),
            *measure_PauliS_in_Z_basis_Q_circ.all_operations(),
        ]
    )
    return full_circuit, Ps, gamma_l


from quchem.Simulating_Quantum_Circuit import *
class VQE_Experiment_Conj_UP():

    def __init__(self, anti_commuting_sets, ansatz_circuit, n_shots, S_key_dict=None):
        self.anti_commuting_sets = anti_commuting_sets
        self.ansatz_circuit = ansatz_circuit
        self.S_key_dict = S_key_dict
        self.n_shots = n_shots

    def Calc_Energy(self):

        E_list = []
        for set_key in self.anti_commuting_sets:

            anti_commuting_set = self.anti_commuting_sets[set_key]

            if len(anti_commuting_set) > 1:

                if self.S_key_dict is None:
                    Q_circuit, Ps, gamma_l = Generate_Full_Q_Circuit_conj(self.ansatz_circuit,
                                                                          anti_commuting_set,
                                                                          0,  # <- S_index set to 0
                                                                          check_reduction=False)

                else:
                    Q_circuit, Ps, gamma_l = Generate_Full_Q_Circuit_conj(self.ansatz_circuit,
                                                                          anti_commuting_set,
                                                                          self.S_key_dict[set_key],
                                                                          # <- S_index set to 0
                                                                          check_reduction=False)

                hist_key_str = Get_Histogram_key(Ps)
                int_state_counter = Simulate_Quantum_Circuit(Q_circuit, self.n_shots, hist_key_str)
                binary_state_counter = Return_as_binary(int_state_counter, hist_key_str)
                exp_result = expectation_value_by_parity(binary_state_counter)

                E_list.append(exp_result * gamma_l)



            else:
                qubitOp = anti_commuting_set[0]

                for PauliWord, const in qubitOp.terms.items():
                    if PauliWord is not ():
                        Q_circuit = Generate_Full_Q_Circuit(self.ansatz_circuit, qubitOp)
                        hist_key_str = Get_Histogram_key(qubitOp)
                        int_state_counter = Simulate_Quantum_Circuit(Q_circuit, self.n_shots, hist_key_str)
                        binary_state_counter = Return_as_binary(int_state_counter, hist_key_str)
                        exp_result = expectation_value_by_parity(binary_state_counter)
                        E_list.append(exp_result * const)

                    else:
                        E_list.append(const)

        return sum(E_list).real

########## Linear Algebra approach

def Generate_Full_Q_Circuit_Conj_NO_M_gates(Full_Ansatz_Q_Circuit, anti_commuting_set, S_index, check_reduction=False):
    """
    Function to build full Q Circuit... ansatz circuit + R_S
    But with NO measurement process!
    Args:
        Full_Ansatz_Q_Circuit (cirq.Circuit): ansatz quantum circuit
        anti_commuting_set(list): list of anti commuting QubitOperators
        S_index(int): index for Ps in anti_commuting_set list
        check_reduction (optional, bool): use linear algebra to check that 𝑅s† 𝐻s 𝑅s == 𝑃s
    returns:
        full_RS_circuit(cirq.Circuit): Q_circuit for R_s operator
        Ps (QubitOperator): Pauli_S operator with cofactor of 1!
        gamma_l (float): normalization term

    """
    Reduction_circuit_circ, Ps, gamma_l = Build_reduction_circuit(anti_commuting_set, S_index,
                                                                  check_reduction=check_reduction)

    full_circuit = cirq.Circuit(
        [
            Full_Ansatz_Q_Circuit.all_operations(),
            *Reduction_circuit_circ.all_operations(),
        ]
    )
    return full_circuit, Ps, gamma_l

class VQE_Experiment_Conj_UP_lin_alg():

    def __init__(self, anti_commuting_sets, ansatz_circuit, S_key_dict=None):
        self.anti_commuting_sets = anti_commuting_sets
        self.ansatz_circuit = ansatz_circuit
        self.S_key_dict = S_key_dict

        input_state = [np.array([[1], [0]]) for _ in range(len(ansatz_circuit.all_qubits()))]
        self.input_ket = reduce(kron, input_state).todense()

    def Calc_Energy(self):

        E_list = []
        for set_key in self.anti_commuting_sets:

            anti_commuting_set = self.anti_commuting_sets[set_key]

            if len(anti_commuting_set) > 1:
                if self.S_key_dict is None:
                    Q_circuit, Ps, gamma_l = Generate_Full_Q_Circuit_Conj_NO_M_gates(self.ansatz_circuit,
                                                                                     anti_commuting_set,
                                                                                     0,  # <- S_index set to 0
                                                                                     check_reduction=False)

                else:
                    Q_circuit, Ps, gamma_l = Generate_Full_Q_Circuit_Conj_NO_M_gates(self.ansatz_circuit,
                                                                                     anti_commuting_set,
                                                                                     self.S_key_dict[set_key],
                                                                                     # ^^^-- S_index defined in dict
                                                                                     check_reduction=False)

                circuit_matrix = Q_circuit.unitary()

                final_state_ket = circuit_matrix.dot(self.input_ket)
                final_state_bra = final_state_ket.transpose().conj()

                Ps_matrix = Get_pauli_matrix(Ps, len(self.ansatz_circuit.all_qubits()))

                exp_result = final_state_bra.dot(Ps_matrix.todense().dot(final_state_ket))
                E_list.append(exp_result.item(0) * gamma_l)

            else:
                qubitOp = anti_commuting_set[0]

                for PauliWord, const in qubitOp.terms.items():
                    if PauliWord is not ():
                        circuit_matrix = self.ansatz_circuit.unitary()

                        final_state_ket = circuit_matrix.dot(self.input_ket)
                        final_state_bra = final_state_ket.transpose().conj()

                        P_matrix = Get_pauli_matrix(qubitOp, len(self.ansatz_circuit.all_qubits()))

                        exp_result = final_state_bra.dot(P_matrix.todense().dot(final_state_ket))
                        E_list.append(exp_result.item(0) * const)

                    else:
                        E_list.append(const)

        return sum(E_list).real


### SeqRot operator new check method ###
from scipy.sparse.linalg import expm
from openfermion import qubit_operator_sparse
def sparse_allclose(A, B, atol=1e-8, rtol=1e-05):
    # https://stackoverflow.com/questions/47770906/how-to-test-if-two-sparse-arrays-are-almost-equal

    if np.array_equal(A.shape, B.shape) == 0:
        raise ValueError('Matrices different shapes!')

    r1, c1, v1 = find(A)  # row indices, column indices, and values of the nonzero matrix entries
    r2, c2, v2 = find(B)

    # take all the important rows and columns
    rows = np.union1d(r1, r2)
    columns = np.union1d(c1, c2)

    A_check = A[rows, columns]
    B_check = B[rows, columns]

    return np.allclose(A_check, B_check, atol=atol, rtol=rtol)


def SeqRot_Check(AC_set, S_index, N_Qubits, atol=1e-8, rtol=1e-05):
    if len(AC_set) < 2:
        raise ValueError('no unitary partitioning possible for set sizes less than 2')

    X_sk_theta_sk_list, full_normalised_set, Ps, gamma_l = Get_Xsk_op_list(AC_set, S_index)

    R_sk_list = []
    for X_sk_Op, theta_sk in X_sk_theta_sk_list:
        pauliword_X_sk_MATRIX = qubit_operator_sparse(QubitOperator(list(X_sk_Op.terms.keys())[0], -1j),
                                                      n_qubits=N_Qubits)
        const_X_sk = list(X_sk_Op.terms.values())[0]

        R_sk_list.append(expm(pauliword_X_sk_MATRIX * theta_sk / 2 * const_X_sk))

    R_S_matrix = reduce(np.dot, R_sk_list[::-1])  # <- note reverse order!

    Ps_mat = qubit_operator_sparse(Ps, n_qubits=N_Qubits)

    H_S = QubitOperator()
    for QubitOp in full_normalised_set['PauliWords']:
        H_S += QubitOp
    H_S_matrix = qubit_operator_sparse(H_S, n_qubits=N_Qubits)

    RHR = R_S_matrix.dot(H_S_matrix.dot(R_S_matrix.conj().transpose()))
    return sparse_allclose(Ps_mat, RHR, atol=atol, rtol=rtol)

from scipy.sparse.linalg import eigs as sparse_eigs
def SeqRot_Energy(AC_set, S_index, N_Qubits, atol=1e-8, rtol=1e-05, check_reduction=False):
    if len(AC_set) < 2:
        raise ValueError('no unitary partitioning possible for set sizes less than 2')

    X_sk_theta_sk_list, full_normalised_set, Ps, gamma_l = Get_Xsk_op_list(AC_set, S_index)

    R_sk_list = []
    for X_sk_Op, theta_sk in X_sk_theta_sk_list:
        pauliword_X_sk_MATRIX = qubit_operator_sparse(QubitOperator(list(X_sk_Op.terms.keys())[0], -1j),
                                                      n_qubits=N_Qubits)
        const_X_sk = list(X_sk_Op.terms.values())[0]

        R_sk_list.append(expm(pauliword_X_sk_MATRIX * theta_sk / 2 * const_X_sk))

    R_S_matrix = reduce(np.dot, R_sk_list[::-1])  # <- note reverse order!

    H_S = QubitOperator()
    for QubitOp in full_normalised_set['PauliWords']:
        H_S += QubitOp
    H_S_matrix = qubit_operator_sparse(H_S, n_qubits=N_Qubits)

    RHR_matrix = R_S_matrix.dot(H_S_matrix.dot(R_S_matrix.conj().transpose()))

    eig_values, eig_vectors = sparse_eigs(RHR_matrix)
    Energy = min(eig_values) * gamma_l

    if check_reduction:
        Ps_mat = qubit_operator_sparse(Ps, n_qubits=N_Qubits)
        check_flag = sparse_allclose(Ps_mat, RHR_matrix, atol=atol, rtol=rtol)
        return Energy, check_flag
    else:
        return Energy