from functools import reduce
from scipy.sparse import csr_matrix
from scipy.sparse import kron
import numpy as np


from openfermion.ops import QubitOperator
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

def Get_X_sk(qubitOp_Ps, qubitOp_Pk):
    """
    Function takes in two QubitOperators (P_s, and P_k) and returns the QubitOperator X_sk (= i P_s P_k)


    Args:
        qubitOp_Ps (QubitOperator): QubitOperator of P_s
        qubitOp_Pk (QubitOperator): QubitOperator of P_k

    Returns:
        X_sk (QubitOperator): QubitOperator of X_sk

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

    PauliStr_tuples_Ps = [tup for PauliStrs, const in qubitOp_Ps.terms.items() for tup in PauliStrs]
    qubitNo_Ps, PauliStr_Ps = zip(*PauliStr_tuples_Ps)
    qubitNo_Ps = np.array(qubitNo_Ps)

    PauliStr_tuples_Pk = [tup for PauliStrs, const in qubitOp_Pk.terms.items() for tup in PauliStrs]
    qubitNo_Pk, PauliStr_Pk = zip(*PauliStr_tuples_Pk)
    qubitNo_Pk = np.array(qubitNo_Pk)

    common_qubits = np.intersect1d(qubitNo_Ps, qubitNo_Pk)

    PauliStr_Ps_common = np.take(PauliStr_Ps, np.where(np.isin(qubitNo_Ps, common_qubits) == True)).flatten()
    PauliStr_Pk_common = np.take(PauliStr_Pk, np.where(np.isin(qubitNo_Pk, common_qubits) == True)).flatten()

    new_paulistr_list = []
    new_factor = []
    for index, pauli_str_Ps in enumerate(PauliStr_Ps_common):

        pauli_str_Pk = PauliStr_Pk_common[index]
        qubitNo = common_qubits[index]

        combined_pauli_str = pauli_str_Ps + pauli_str_Pk

        if convert_term[combined_pauli_str][1] != 'I':
            new_pauli_str = convert_term[combined_pauli_str][1] + str(qubitNo)
            new_paulistr_list.append(new_pauli_str)

            new_factor.append(convert_term[combined_pauli_str][0])

    new_constant = np.prod(new_factor)

    for index, qubitNo in enumerate(qubitNo_Ps):
        if qubitNo not in common_qubits:
            Paulistring_Ps = PauliStr_Ps[index]
            new_paulistr_list.append(Paulistring_Ps + str(qubitNo))

    for index, qubitNo in enumerate(qubitNo_Pk):
        if qubitNo not in common_qubits:
            Paulistring_Pk = PauliStr_Pk[index]
            new_paulistr_list.append(Paulistring_Pk + str(qubitNo))

    seperator = ' '
    pauliStr_list = seperator.join(new_paulistr_list)

    X_sk = QubitOperator(pauliStr_list, 1j * new_constant)

    return X_sk

def Get_X_sk_operators(normalised_anticommuting_set_DICT, S=0):  #
    """

TODO

    """

    qubit_Op_list_normalisted = normalised_anticommuting_set_DICT['PauliWords'].copy()

    if len(qubit_Op_list_normalisted) > 1:

        PauliS = qubit_Op_list_normalisted.pop(S)
        beta_S = list(PauliS.terms.values())[0]

        Op_list = []
        running_total = 0
        for index, PauliK in enumerate(qubit_Op_list_normalisted):
            if index == 0:
                X_sk_op = Get_X_sk(PauliS, PauliK)
                beta_K = list(PauliK.terms.values())[0]
                theta_sk = np.arctan(beta_K / beta_S)

                if beta_S.real < 0:
                    theta_sk = theta_sk + np.pi

            else:
                beta_K = list(PauliK.terms.values())[0]
                running_total += beta_K ** 2
                theta_sk = np.arctan(beta_K / np.sqrt(beta_S ** 2 + running_total))

            Op_list.append({'X_sk': X_sk_op,
                            'theta_sk_over2': theta_sk / 2})  # , 'factor': normalised_anti_commuting_sets[key]['factor']})

        #         if beta_S<0: # if list(PauliS.terms.values())[0] < 0:
        #             sign_gamma_l = -1
        #         else:
        #             sign_gamma_l = 1

        return {'X_sk_and_theta_terms': Op_list, 'PauliWord_S': QubitOperator(list(PauliS.terms.keys())[0], 1),
                'gamma_l': normalised_anticommuting_set_DICT['gamma_l']}  # *sign_gamma_l}

# ANDREW CODE:
# def thetasFromOplist(normalisedOplist):
#     betas = [x for x in normalisedOplist]
#     squaredBetas = [x**2 for x in betas]
#
#     runningTotal = squaredBetas[-1]
#     squaredBetaSums = [runningTotal]
#     for i in range(1,len(normalisedOplist)-1):
#         runningTotal += squaredBetas[i-1]
#         squaredBetaSums.append(runningTotal)
#
#     l2Betas = [x**(1./2.) for x in squaredBetaSums]
#     l2Betas[0] = betas[-1]
#     thetas = [np.arctan(betas[i]/l2Betas[i]) for i in range(len(l2Betas))]
#     if betas[-1].real < 0.:
#         thetas[0] = thetas[0] + np.pi
#     return thetas
#
# thetasFromOplist([0.9668047296891765, -0.25551636865500044])
from quchem.quantum_circuit_functions import full_exponentiated_PauliWord_circuit
from openfermion.ops import QubitOperator
import cirq

def Build_reduction_circuit(normalised_anticommuting_set_X_sk_DICT):
    for term in normalised_anticommuting_set_X_sk_DICT['X_sk_and_theta_terms']:
        pauliword_X_sk = list(term['X_sk'].terms.keys())[0]
        const_X_sk = list(term['X_sk'].terms.values())[0]

        theta_sk_over2 = term['theta_sk_over2']

        full_exp_circ_obj = full_exponentiated_PauliWord_circuit(QubitOperator(pauliword_X_sk, -1j),
                                                                 theta_sk_over2 * const_X_sk)

        circuit = cirq.Circuit(
            cirq.decompose_once((full_exp_circ_obj(*cirq.LineQubit.range(full_exp_circ_obj.num_qubits())))))

        yield circuit

from quchem.quantum_circuit_functions import change_pauliword_to_Z_basis_then_measure
def Generate_Full_Q_Circuit_unitary_part(Full_Ansatz_Q_Circuit, normalised_anticommuting_set_X_sk_DICT):
    """
     TODO

    Args:


    Returns:
        full_circuit (cirq.circuits.circuit.Circuit): Full cirq VQE circuit

    """

    Reduction_circuit_obj = Build_reduction_circuit(normalised_anticommuting_set_X_sk_DICT)
    Reduction_circuit_circ = cirq.Circuit(Reduction_circuit_obj)

    measure_PauliS_in_Z_basis_obj = change_pauliword_to_Z_basis_then_measure(
        normalised_anticommuting_set_X_sk_DICT['PauliWord_S'])
    measure_PauliS_in_Z_basis_Q_circ = cirq.Circuit(cirq.decompose_once(
        (measure_PauliS_in_Z_basis_obj(*cirq.LineQubit.range(measure_PauliS_in_Z_basis_obj.num_qubits())))))

    full_circuit = cirq.Circuit(
        [
            Full_Ansatz_Q_Circuit.all_operations(),
            *Reduction_circuit_circ.all_operations(),
            *measure_PauliS_in_Z_basis_Q_circ.all_operations(),
        ]
    )
    return full_circuit

from quchem.Simulating_Quantum_Circuit import *
class VQE_Experiment_UP():
    def __init__(self, graph_dict_sets, ansatz_circuit, n_shots, S_key_dict=None):
        self.graph_dict_sets = graph_dict_sets
        self.ansatz_circuit = ansatz_circuit
        self.S_key_dict = S_key_dict
        self.n_shots = n_shots

    def Calc_Energy(self):

        E_list = []
        for set_key in self.graph_dict_sets:

            if len(self.graph_dict_sets[set_key]) > 1:

                normalised_set = Get_beta_j_cofactors(self.graph_dict_sets[set_key])

                if self.S_key_dict is None:
                    X_sk_dict = Get_X_sk_operators(normalised_set, S=0)
                else:
                    X_sk_dict = Get_X_sk_operators(normalised_set, S=self.S_key_dict[set_key])

                Q_circuit = Generate_Full_Q_Circuit_unitary_part(self.ansatz_circuit, X_sk_dict)
                hist_key_str = Get_Histogram_key(X_sk_dict['PauliWord_S'])
                int_state_counter = Simulate_Quantum_Circuit(Q_circuit, self.n_shots, hist_key_str)
                binary_state_counter = Return_as_binary(int_state_counter, hist_key_str)
                exp_result = expectation_value_by_parity(binary_state_counter)
                E_list.append(exp_result * X_sk_dict['gamma_l'])

            #                 print('')
            #                 print('PauliWord_S = ',X_sk_dict['PauliWord_S'])
            #                 print(exp_result, X_sk_dict['gamma_l'])
            #                 print('')

            else:
                qubitOp = self.graph_dict_sets[set_key][0]

                for PauliWord, const in qubitOp.terms.items():

                    #                     print('')
                    #                     print(qubitOp)

                    if PauliWord is not ():
                        Q_circuit = Generate_Full_Q_Circuit(self.ansatz_circuit, qubitOp)
                        hist_key_str = Get_Histogram_key(qubitOp)
                        int_state_counter = Simulate_Quantum_Circuit(Q_circuit, self.n_shots, hist_key_str)
                        binary_state_counter = Return_as_binary(int_state_counter, hist_key_str)
                        exp_result = expectation_value_by_parity(binary_state_counter)
                        E_list.append(exp_result * const)

                    #                         print(exp_result, const)
                    #                         print('')

                    else:
                        E_list.append(const)

        #                         print(const)
        #                         print('')

        return sum(E_list).real

    def Get_wavefunction_of_state(self, sig_figs=3):
        return Get_wavefunction(self.ansatz_circuit, sig_figs=sig_figs)


# lin alg approach:

def Generate_Full_Q_Circuit_unitary_part_NO_M_gates(Full_Ansatz_Q_Circuit, normalised_anticommuting_set_X_sk_DICT):
    """
     TODO

    Args:


    Returns:
        full_circuit (cirq.circuits.circuit.Circuit): Full cirq VQE circuit

    """

    Reduction_circuit_obj = Build_reduction_circuit(normalised_anticommuting_set_X_sk_DICT)
    Reduction_circuit_circ = cirq.Circuit(Reduction_circuit_obj)

    full_circuit = cirq.Circuit(
        [
            Full_Ansatz_Q_Circuit.all_operations(),
            *Reduction_circuit_circ.all_operations(),
        ]
    )
    return full_circuit
class VQE_Experiment_UP_lin_alg():
    def __init__(self, graph_dict_sets, ansatz_circuit,N_system_qubits, S_key_dict=None):
        self.graph_dict_sets = graph_dict_sets
        self.ansatz_circuit = ansatz_circuit
        self.S_key_dict = S_key_dict
        self.N_system_qubits = N_system_qubits

        self.pauliDict = {'X': np.array([[0, 1], [1, 0]]),
                          'Y': np.array([[0, -1j], [1j, 0]]),
                          'Z': np.array([[1, 0], [0, -1]]),
                          'I': np.eye(2)}

        self.zero_state = np.array([[1], [0]])

    def Get_output_ket(self, Q_circuit_no_M_gates):

        input_state = [self.zero_state for _ in range(len(Q_circuit_no_M_gates.all_qubits()))]
        input_ket = reduce(kron, input_state)
        circuit_matrix = Q_circuit_no_M_gates.unitary()

        output_ket = circuit_matrix.dot(input_ket.todense())

        if not np.isclose(sum([i**2 for i in output_ket]), 1):
            raise ValueError('output ket is not normalised properly')

        return np.array(output_ket) #.reshape([(2 ** len(self.ansatz_circuit.all_qubits())), 1])

    def Get_pauli_matrix(self, PauliOp):

        list_Q_nos, list_P_strs = list(zip(*[Paulistrs for Paulistrs, const in PauliOp.terms.items()][0]))

        list_of_ops = []
        for i in range(self.N_system_qubits):
            if i not in list_Q_nos:
                list_of_ops.append(self.pauliDict['I'])
            else:
                index = list_Q_nos.index(i)
                list_of_ops.append(self.pauliDict[list_P_strs[index]])
        matrix = reduce(kron, list_of_ops)

        return matrix

    def Calc_Energy(self):

        E_list = []
        for set_key in self.graph_dict_sets:

            if len(self.graph_dict_sets[set_key]) > 1:

                normalised_set = Get_beta_j_cofactors(self.graph_dict_sets[set_key])

                if self.S_key_dict is None:
                    X_sk_dict = Get_X_sk_operators(normalised_set, S=0)
                else:
                    X_sk_dict = Get_X_sk_operators(normalised_set, S=self.S_key_dict[set_key])

                Q_circuit = Generate_Full_Q_Circuit_unitary_part_NO_M_gates(self.ansatz_circuit, X_sk_dict)

                state_ket = self.Get_output_ket(Q_circuit)
                state_bra = state_ket.transpose().conj()
                H_sub_term_matrix = self.Get_pauli_matrix(X_sk_dict['PauliWord_S'])

                energy = state_bra.dot(H_sub_term_matrix.dot(state_ket))
                E_list.append(energy[0][0] * X_sk_dict['gamma_l'])

            else:
                single_PauliOp = self.graph_dict_sets[set_key][0]

                if list(single_PauliOp.terms.keys())[0] == ():
                    E_list.append(list(single_PauliOp.terms.values())[0])
                else:
                    state_ket = self.Get_output_ket(self.ansatz_circuit)
                    state_bra = state_ket.transpose().conj()
                    # H_sub_term_matrix = get_sparse_operator(single_PauliOp, n_qubits=self.N_system_qubits)
                    H_sub_term_matrix = self.Get_pauli_matrix(single_PauliOp)
                    energy = state_bra.dot(H_sub_term_matrix.dot(state_ket))
                    E_list.append(energy[0][0] * list(single_PauliOp.terms.values())[0])

        return sum(E_list).real

    def Get_wavefunction_of_state(self, sig_figs=3):
        return Get_wavefunction(self.ansatz_circuit, sig_figs=sig_figs)

