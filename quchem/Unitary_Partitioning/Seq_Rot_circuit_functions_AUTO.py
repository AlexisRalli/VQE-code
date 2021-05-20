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



### automated choice of term to reduce too and circuit depth reduced
from quchem.Qcircuit.Optimized_exp_pauliword_circuit_functions import Optimized_LADDER_circuit
from quchem.Misc_functions.Misc_functions import choose_Pn_index, lexicographical_sort_BASIS_MATCH, lexicographical_sort_LADDER_CNOT_cancel
from copy import deepcopy as copy
def Auto_Build_R_SeqRot_Q_circuit_manual_Reduced(anti_commuting_set, N_Qubits, check_reduction_lin_alg=False, atol=1e-8, rtol=1e-05, check_circuit=False, maximise_CNOT_reduction=True):
    """
    Function to build R_S where S_index has been chosen automatically - see ```choose_Pn_index``` function. A lexicographical sort has also been used to optimize
    number of cancellations in rotation circuits. Here change of basis single qubit gates have been targeted

    Args:
        anti_commuting_set(list): list of anti commuting QubitOperators
        N_Qubits (int): number of qubits
        check_reduction_lin_alg (optional, bool): use linear algebra to check that 𝑅s† 𝐻s 𝑅s == 𝑃s
        check_circuit (bool): check if circuit unitary performs unitary partitioning correctly via lin alg on anti-commuting subset
        maximise_CNOT_reduction (bool): whether to order terms in set to maxmise change of basis or CNOT cancellations
    returns:
        full_RS_circuit(cirq.Circuit): Q_circuit for R_s operator
        Ps (QubitOperator): Pauli_S operator with cofactor of 1!
        gamma_l (float): normalization term

    """
    anti_commuting_set = copy(anti_commuting_set)
    Ps_index = choose_Pn_index(anti_commuting_set)
    B_Ps = anti_commuting_set.pop(Ps_index) # REMOVE BETA_S_P_S from AC set
    
    if maximise_CNOT_reduction:
        re_orded_AC_set = lexicographical_sort_LADDER_CNOT_cancel(anti_commuting_set)
    else:
        re_orded_AC_set = lexicographical_sort_BASIS_MATCH(anti_commuting_set)

    re_orded_AC_set.append(B_Ps) # add Ps back in at last index!
    S_index =-1 # hence s index is last term in list
    
    
    X_sk_theta_sk_list, full_normalised_set, Ps, gamma_l = Get_Xsk_op_list(re_orded_AC_set,
                                                                            S_index, 
                                                                            N_Qubits,
                                                                         check_reduction=check_reduction_lin_alg,
                                                                          atol=atol, 
                                                                          rtol=rtol)
    
    angle_list=[]
    op_list =[]
    for X_sk_Op, theta_sk in X_sk_theta_sk_list:
        pauliword_X_sk = list(X_sk_Op.terms.keys())[0]
        const_X_sk = list(X_sk_Op.terms.values())[0]
        
        
        angle_list.append(theta_sk / 2 * const_X_sk)
        
        op_list.append(QubitOperator(pauliword_X_sk, -1j))
    

    full_RS_circuit = Optimized_LADDER_circuit(op_list, angle_list, check_reduction=check_circuit)
    
    if check_circuit:

        H_S = QubitOperator()
        for QubitOp in full_normalised_set['PauliWords']:
            H_S += QubitOp
        H_S_matrix = qubit_operator_sparse(H_S, n_qubits=N_Qubits)
        
        qbits = cirq.LineQubit.range(N_Qubits)
        R_S_matrix = full_RS_circuit.unitary(qubits_that_should_be_present=qbits)

        Ps_mat=qubit_operator_sparse(Ps, n_qubits=N_Qubits)
        reduction_mat = R_S_matrix.dot(H_S_matrix.dot(R_S_matrix.conj().transpose()))

        if not sparse_allclose(Ps_mat, reduction_mat):
            print('reduction circuit incorrect...   𝑅s 𝐻s 𝑅s† != 𝑃s')

    return full_RS_circuit, Ps, gamma_l

from quchem.Misc_functions.Misc_functions import optimized_cirq_circuit_IBM_compiler
def Auto_Build_R_SeqRot_Q_circuit_IBM_Reduced(anti_commuting_set, N_Qubits, check_reduction_lin_alg=False, atol=1e-8, rtol=1e-05,
 check_circuit=False, maximise_CNOT_reduction=True, allowed_qiskit_gates=['id', 'rz', 'ry', 'rx', 'cx' ,'s', 'h', 'y','z'],
 IBM_opt_level=3):
    """
    Function to build R_S where S_index has been chosen automatically - see ```choose_Pn_index``` function. A lexicographical sort has also been used to optimize
    number of cancellations in rotation circuits. IBM's compiler is used to reduce the circuit depth, NOTE global phase difference NOT enforced.

    Args:
        anti_commuting_set(list): list of anti commuting QubitOperators
        N_Qubits (int): number of qubits
        check_reduction_lin_alg (optional, bool): use linear algebra to check that 𝑅s† 𝐻s 𝑅s == 𝑃s
        check_circuit (bool): check if circuit unitary performs unitary partitioning correctly via lin alg on anti-commuting subset
        maximise_CNOT_reduction (bool): whether to order terms in set to maxmise change of basis or CNOT cancellations
        IBM_opt_level (int): optimization level of IBM compiler (see optimized_cirq_circuit_IBM_compiler for further details)
    Returns:
        full_RS_circuit(cirq.Circuit): Q_circuit for R_s operator
        Ps (QubitOperator): Pauli_S operator with cofactor of 1!
        gamma_l (float): normalization term

    """
    anti_commuting_set = copy(anti_commuting_set)
    Ps_index = choose_Pn_index(anti_commuting_set)
    B_Ps = anti_commuting_set.pop(Ps_index) # REMOVE BETA_S_P_S from AC set
    
    if maximise_CNOT_reduction:
        re_orded_AC_set = lexicographical_sort_LADDER_CNOT_cancel(anti_commuting_set)
    else:
        re_orded_AC_set = lexicographical_sort_BASIS_MATCH(anti_commuting_set)

    re_orded_AC_set.append(B_Ps) # add Ps back in at last index!
    S_index =-1 # hence s index is last term in list
    
    
    X_sk_theta_sk_list, full_normalised_set, Ps, gamma_l = Get_Xsk_op_list(re_orded_AC_set,
                                                                            S_index, 
                                                                            N_Qubits,
                                                                         check_reduction=check_reduction_lin_alg,
                                                                          atol=atol, 
                                                                          rtol=rtol)
    
    full_RS_circuit = cirq.Circuit()
    for X_sk_Op, theta_sk in X_sk_theta_sk_list:
        pauliword_X_sk = list(X_sk_Op.terms.keys())[0]
        const_X_sk = list(X_sk_Op.terms.values())[0]

        full_exp_circ_obj = full_exponentiated_PauliWord_circuit(QubitOperator(pauliword_X_sk, -1j),
                                                                 theta_sk / 2 * const_X_sk)

        circuit = cirq.Circuit(
            cirq.decompose_once((full_exp_circ_obj(*cirq.LineQubit.range(full_exp_circ_obj.num_qubits())))))

        full_RS_circuit.append(circuit)


    # IBM re-compile
    full_RS_circuit, phase, Zpowgate_flag = optimized_cirq_circuit_IBM_compiler(
                                full_RS_circuit,
                                opt_level=IBM_opt_level,
                                allowed_gates=allowed_qiskit_gates,
                                check_optimization = True)

    if check_circuit:

        H_S = QubitOperator()
        for QubitOp in full_normalised_set['PauliWords']:
            H_S += QubitOp
        H_S_matrix = qubit_operator_sparse(H_S, n_qubits=N_Qubits)
        
        # qbits = [cirq.NamedQubit(f'q_{i}') for i in range(N_Qubits)] # IBM compiler causes linequbits to change to Namedqubits
        qbits = cirq.LineQubit.range(N_Qubits)
        R_S_matrix = full_RS_circuit.unitary(qubits_that_should_be_present=qbits)

        Ps_mat=qubit_operator_sparse(Ps, n_qubits=N_Qubits)
        reduction_mat = R_S_matrix.dot(H_S_matrix.dot(R_S_matrix.conj().transpose()))

        if not sparse_allclose(Ps_mat, reduction_mat):
            print('reduction circuit incorrect...   𝑅s 𝐻s 𝑅s† != 𝑃s')
            fro_norm = np.linalg.norm(Ps_mat.todense()-reduction_mat, ord='fro')
            print(f'frobius norm between R @ H @ R† and Ps: {fro_norm}')

    return full_RS_circuit, Ps, gamma_l


def Full_SeqRot_auto_Rl_Circuit_manual_Reduced(Full_Ansatz_Q_Circuit, anti_commuting_set, N_Qubits, check_reduction_lin_alg=False, check_circuit=False, maximise_CNOT_reduction=True):
    """
    Function to build full Q Circuit... ansatz circuit + R_S
    Here choice of Pauli_S has been automated and circuit has been reduced manually

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
    Reduction_circuit_circ, Ps, gamma_l = Auto_Build_R_SeqRot_Q_circuit_manual_Reduced(
                                                            anti_commuting_set,
                                                            N_Qubits, 
                                                            check_reduction_lin_alg=check_reduction_lin_alg, 
                                                            atol=1e-8, 
                                                            rtol=1e-05, 
                                                            check_circuit=check_circuit,
                                                             maximise_CNOT_reduction=maximise_CNOT_reduction)

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

def Full_SeqRot_auto_Rl_Circuit_IBM_Reduced(Full_Ansatz_Q_Circuit, anti_commuting_set, N_Qubits, check_reduction_lin_alg=False, check_circuit=False, 
                                            maximise_CNOT_reduction=True, allowed_qiskit_gates=['id', 'rz', 'ry', 'rx', 'cx' ,'s', 'h', 'y','z'], IBM_opt_level=3):
    """
    Function to build full Q Circuit... ansatz circuit + R_S
    Here choice of Pauli_S has been automated and circuit has been reduced using IBM compiler

    Args:
        Full_Ansatz_Q_Circuit (cirq.Circuit): ansatz quantum circuit
        anti_commuting_set(list): list of anti commuting QubitOperators
        S_index(int): index for Ps in anti_commuting_set list
        check_reduction (optional, bool): use linear algebra to check that 𝑅s† 𝐻s 𝑅s == 𝑃s
        IBM_opt_level (int): optimization level of IBM compiler (see optimized_cirq_circuit_IBM_compiler for further details)
    returns:
        full_RS_circuit(cirq.Circuit): Q_circuit for R_s operator
        Ps (QubitOperator): Pauli_S operator with cofactor of 1!
        gamma_l (float): normalization term

    """
    Reduction_circuit_circ, Ps, gamma_l = Auto_Build_R_SeqRot_Q_circuit_IBM_Reduced(
                                                            anti_commuting_set,
                                                            N_Qubits, 
                                                            check_reduction_lin_alg=check_reduction_lin_alg, 
                                                            atol=1e-8, 
                                                            rtol=1e-05, 
                                                            check_circuit=check_circuit,
                                                             maximise_CNOT_reduction=maximise_CNOT_reduction,
                                                             allowed_qiskit_gates=allowed_qiskit_gates,
                                                             IBM_opt_level=IBM_opt_level)

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



########## Linear Algebra auto choice of Ps circuit approach

class Auto_Seq_Rot_VQE_Experiment_UP_manual_reduced_circuit_lin_alg():

    def __init__(self, anti_commuting_sets, ansatz_circuit):
        self.anti_commuting_sets = anti_commuting_sets
        self.ansatz_circuit = ansatz_circuit
        self.n_qubits = len(ansatz_circuit.all_qubits())

        ansatz_vector = ansatz_circuit.final_state_vector(ignore_terminal_measurements=True)#.reshape((2**self.n_qubits,1))
        self.ansatz_density_mat = np.outer(ansatz_vector, ansatz_vector)

    def Calc_Energy(self, check_reduction_lin_alg=False, check_circuit=False, maximise_CNOT_reduction=True):

        E_list = []
        for set_key in self.anti_commuting_sets:

            anti_commuting_set = self.anti_commuting_sets[set_key]

            if len(anti_commuting_set) > 1:

                Q_circuit, Ps, gamma_l = Full_SeqRot_auto_Rl_Circuit_manual_Reduced(
                                                                                    self.ansatz_circuit, 
                                                                                    anti_commuting_set, 
                                                                                    self.n_qubits, 
                                                                                    check_reduction_lin_alg=check_reduction_lin_alg, 
                                                                                    check_circuit=check_circuit, 
                                                                                    maximise_CNOT_reduction=maximise_CNOT_reduction)

                
                final_state_ket = Q_circuit.final_state_vector(ignore_terminal_measurements=True)#.reshape((2**self.n_qubits,1))

                # note Q_circuit HAS change of basis for Ps! hence measure Z op version now
                PauliStr_Ps, beta_S = tuple(*Ps.terms.items())
                PauliStr_Ps_Z = [(qNo, 'Z')for qNo, Pstr in PauliStr_Ps]
                Ps_Zchange = QubitOperator(PauliStr_Ps_Z, beta_S)

                Ps_matrix = qubit_operator_sparse(Ps_Zchange, n_qubits=self.n_qubits)
                # exp_result = np.trace(np.outer(final_state_ket, final_state_ket)@Ps_matrix)
                # E_list.append(exp_result * gamma_l)

                exp_result = final_state_ket.conj().T @ Ps_matrix @ final_state_ket
                E_list.append(exp_result.item(0) * gamma_l)

            else:
                qubitOp = anti_commuting_set[0]
                P_matrix = qubit_operator_sparse(qubitOp, n_qubits=self.n_qubits)
                exp_result = np.trace(self.ansatz_density_mat@P_matrix)
                E_list.append(exp_result)

        return sum(E_list).real


class Auto_Seq_Rot_VQE_Experiment_UP_IBM_reduced_circuit_lin_alg():

    def __init__(self, anti_commuting_sets, ansatz_circuit, 
        allowed_qiskit_gates=['id', 'rz', 'ry', 'rx', 'cx' ,'s', 'h', 'y','z'],
        IBM_opt_lvl=3):
        self.anti_commuting_sets = anti_commuting_sets
        self.ansatz_circuit = ansatz_circuit
        self.n_qubits = len(ansatz_circuit.all_qubits())

        self.allowed_qiskit_gates = allowed_qiskit_gates
        self.IBM_opt_lvl = IBM_opt_lvl

        ansatz_vector = ansatz_circuit.final_state_vector(ignore_terminal_measurements=True)#.reshape((2**self.n_qubits,1))
        self.ansatz_density_mat = np.outer(ansatz_vector, ansatz_vector)

    def Calc_Energy(self, check_reduction_lin_alg=False, check_circuit=False, maximise_CNOT_reduction=True):

        E_list = []
        for set_key in self.anti_commuting_sets:

            anti_commuting_set = self.anti_commuting_sets[set_key]

            if len(anti_commuting_set) > 1:

                Q_circuit, Ps, gamma_l = Full_SeqRot_auto_Rl_Circuit_IBM_Reduced(
                                                                                    self.ansatz_circuit, 
                                                                                    anti_commuting_set, 
                                                                                    self.n_qubits, 
                                                                                    check_reduction_lin_alg=check_reduction_lin_alg, 
                                                                                    check_circuit=check_circuit, 
                                                                                    maximise_CNOT_reduction=maximise_CNOT_reduction, 
                                                                                    allowed_qiskit_gates=self.allowed_qiskit_gates,
                                                                                    IBM_opt_level=self.IBM_opt_lvl)
                
                final_state_ket = Q_circuit.final_state_vector(ignore_terminal_measurements=True)#.reshape((2**self.n_qubits,1))

                # note Q_circuit HAS change of basis for Ps! hence measure Z op version now
                PauliStr_Ps, beta_S = tuple(*Ps.terms.items())
                PauliStr_Ps_Z = [(qNo, 'Z')for qNo, Pstr in PauliStr_Ps]
                Ps = QubitOperator(PauliStr_Ps_Z, beta_S)

                Ps_matrix = qubit_operator_sparse(Ps, n_qubits=self.n_qubits)
                # exp_result = np.trace(np.outer(final_state_ket, final_state_ket)@Ps_matrix)
                # E_list.append(exp_result * gamma_l)

                exp_result = final_state_ket.conj().T @ Ps_matrix @ final_state_ket
                E_list.append(exp_result.item(0) * gamma_l)

            else:
                qubitOp = anti_commuting_set[0]
                P_matrix = qubit_operator_sparse(qubitOp, n_qubits=self.n_qubits)
                exp_result = np.trace(self.ansatz_density_mat@P_matrix)
                E_list.append(exp_result)

        return sum(E_list).real
