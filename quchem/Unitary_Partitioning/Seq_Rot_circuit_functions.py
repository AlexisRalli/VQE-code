from functools import reduce
from scipy.sparse import csr_matrix
from scipy.sparse import kron
import numpy as np
import cirq

from openfermion.linalg import qubit_operator_sparse
from openfermion.ops import QubitOperator

from quchem.Qcircuit.Ansatz_quantum_circuit_functions import full_exponentiated_PauliWord_circuit
from quchem.Qcircuit.Hamiltonian_term_measurement_functions import change_pauliword_to_Z_basis_then_measure
from quchem.Misc_functions.Misc_functions import sparse_allclose

from quchem.Unitary_Partitioning.Unitary_partitioning_Seq_Rot import Get_Xsk_op_list

def Build_R_SeqRot_Q_circuit(anti_commuting_set, S_index,N_Qubits, check_reduction_lin_alg=False, atol=1e-8, rtol=1e-05, check_circuit=False):
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
    X_sk_theta_sk_list, full_normalised_set, Ps, gamma_l = Get_Xsk_op_list(anti_commuting_set,
                                                                            S_index, 
                                                                            N_Qubits,
                                                                             check_reduction=check_reduction_lin_alg,
                                                                              atol=atol, 
                                                                              rtol=rtol)

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

    if check_circuit:

        H_S = QubitOperator()
        for QubitOp in full_normalised_set['PauliWords']:
            H_S += QubitOp
        H_S_matrix = qubit_operator_sparse(H_S)

        qbits = cirq.LineQubit.range(N_Qubits)

        R_S_matrix = full_RS_circuit.unitary(qubits_that_should_be_present=qbits)

        Ps_mat=qubit_operator_sparse(Ps, n_qubits=N_Qubits)
        reduction_mat = R_S_matrix.dot(H_S_matrix.dot(R_S_matrix.conj().transpose()))

        if not sparse_allclose(Ps_mat, reduction_mat):
            print('reduction circuit incorrect...   𝑅s 𝐻s 𝑅s† != 𝑃s')

    return full_RS_circuit, Ps, gamma_l


def Full_SeqRot_Rl_Circuit(Full_Ansatz_Q_Circuit, anti_commuting_set, S_index, N_Qubits, check_reduction_lin_alg=False):
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
    Reduction_circuit_circ, Ps, gamma_l = Build_R_SeqRot_Q_circuit(anti_commuting_set, S_index, N_Qubits,
                                                                  check_reduction_lin_alg=check_reduction_lin_alg)

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


########## Linear Algebra circuit approach

class Seq_Rot_VQE_Experiment_UP_circuit_lin_alg():

    def __init__(self, anti_commuting_sets, ansatz_circuit, S_key_dict=None):
        self.anti_commuting_sets = anti_commuting_sets
        self.ansatz_circuit = ansatz_circuit
        self.S_key_dict = S_key_dict
        self.n_qubits = len(ansatz_circuit.all_qubits())

        ansatz_vector = ansatz_circuit.final_state_vector(ignore_terminal_measurements=True)#.reshape((2**self.n_qubits,1))
        self.ansatz_density_mat = np.outer(ansatz_vector, ansatz_vector)

    def Calc_Energy(self, check_reduction_lin_alg=False, atol=1e-8, rtol=1e-05, check_circuit=False):

        E_list = []
        for set_key in self.anti_commuting_sets:

            anti_commuting_set = self.anti_commuting_sets[set_key]

            if len(anti_commuting_set) > 1:

                if self.S_key_dict is None:
                    full_RS_circuit, Ps, gamma_l= Build_R_SeqRot_Q_circuit(anti_commuting_set, 
                                                                            0,  # <- S_index set to 0 
                                                                            self.n_qubits,
                                                                            check_reduction_lin_alg=check_reduction_lin_alg, atol=atol, rtol=rtol, check_circuit=check_circuit)

                else:
                    full_RS_circuit, Ps, gamma_l= Build_R_SeqRot_Q_circuit(anti_commuting_set, 
                                                        self.S_key_dict[set_key],
                                                        self.n_qubits,
                                                        check_reduction_lin_alg=check_reduction_lin_alg, atol=atol, rtol=rtol, check_circuit=check_circuit)

                # note Build_R_SeqRot_Q_circuit doesn't use a change of basis for Ps!
                Q_circuit = cirq.Circuit(
                                [
                                    self.ansatz_circuit.all_operations(),
                                    *full_RS_circuit.all_operations(),
                                ]
                            )


                final_state_ket = (Q_circuit.final_state_vector(ignore_terminal_measurements=True)).reshape((2**self.n_qubits,1))
                denisty_mat = np.outer(final_state_ket, final_state_ket)

                Ps_matrix = qubit_operator_sparse(Ps, n_qubits=self.n_qubits)

                exp_result = np.trace(denisty_mat@Ps_matrix)
                E_list.append(exp_result*gamma_l)

            else:
                qubitOp = anti_commuting_set[0]
                P_matrix = qubit_operator_sparse(qubitOp, n_qubits=self.n_qubits)
                exp_result = np.trace(self.ansatz_density_mat@P_matrix)
                E_list.append(exp_result)

        return sum(E_list).real


########## sampling Quantum Circuit
class Seq_Rot_VQE_Experiment_UP_circuit_sampling():

    # TODO: currently changed functions - NO LONGER WORKING!

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
                    Q_circuit, Ps, gamma_l = Generate_Ansatz_SeqRot_R_Q_Circuit(self.ansatz_circuit,
                                                                          anti_commuting_set,
                                                                          0,  # <- S_index set to 0
                                                                          check_reduction=False)

                else:
                    Q_circuit, Ps, gamma_l = Generate_Ansatz_SeqRot_R_Q_Circuit(self.ansatz_circuit,
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
                    if PauliWord != ():
                        Q_circuit = Generate_Full_Q_Circuit(self.ansatz_circuit, qubitOp)
                        hist_key_str = Get_Histogram_key(qubitOp)
                        int_state_counter = Simulate_Quantum_Circuit(Q_circuit, self.n_shots, hist_key_str)
                        binary_state_counter = Return_as_binary(int_state_counter, hist_key_str)
                        exp_result = expectation_value_by_parity(binary_state_counter)
                        E_list.append(exp_result * const)

                    else:
                        E_list.append(const)

        return sum(E_list).real