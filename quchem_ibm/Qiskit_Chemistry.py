import cirq
import numpy as np
from qiskit.quantum_info import Operator
from qiskit import QuantumCircuit, QuantumRegister, execute
from qiskit import Aer
import qiskit
from openfermion.transforms import get_sparse_operator
from qiskit.extensions import UnitaryGate
from tqdm import tqdm
import pickle
import os
import datetime

def Total_decompose_qiskit(qc):
    while True:
        qc_d = qc.decompose()
        if qc_d == qc:
            return qc_d
        else:
            qc = qc_d

def HF_state_IBM_circuit(HF_state, q_register, q_circuit):
    ## re-order IN REVERSE!!!!!!!!!!!!!! IMPORTANT!!!!!

    for qNo, bit in enumerate(HF_state):
        if bit == 1:
            q_circuit.x(q_register[qNo])
        elif bit == 0:
            continue
        else:
            raise ValueError('HF state not binary: {}'.format(HF_state))

    return q_circuit

def My_Rz_gate(theta):
    unitary_matrix = cirq.rz(theta)._unitary_()

    return UnitaryGate(unitary_matrix, label='My_Rz({})'.format(np.around(theta, 3)))

def exp_PauliWord_circuit_IBM(PauliWord, theta, q_register, q_circuit):
    q_circuit = q_circuit.copy()
    qubitNos, PauliStrs = zip(*list(*PauliWord.terms.keys()))

    control_qubit = max(qubitNos)
    min_qubit = min(qubitNos)

    # change basis
    for index, qNo in enumerate(qubitNos):
        Pstr = PauliStrs[index]
        qNo = int(qNo)
        if Pstr == 'X':
            q_circuit.h(q_register[qNo])
        elif Pstr == 'Y':
            q_circuit.rx((+np.pi / 2), q_register[qNo])
        elif Pstr == 'Z':
            continue
        else:
            raise ValueError('Not a PauliWord')

    # entangle
    for index, qNo in enumerate(qubitNos):
        Pstr = PauliStrs[index]
        qNo = int(qNo)
        if qNo < control_qubit:
            next_qubit = int(qubitNos[index + 1])
            q_circuit.cx(q_register[qNo], q_register[next_qubit])

    # rz
    for index, qNo in enumerate(qubitNos):
        qNo = int(qNo)
        if qNo == control_qubit:
            cofactor = list(PauliWord.terms.values())[0]

            if isinstance(cofactor, complex):
                if cofactor.imag < 0:
                    Rzgate = My_Rz_gate((2 * theta * np.abs(cofactor.imag)).real)
                    q_circuit.append(Rzgate, [control_qubit])
                #                     q_circuit.rz((2 * theta * np.abs(cofactor.imag)).real, q_register[control_qubit])
                else:
                    # times angle by negative one to get implementation
                    Rzgate = My_Rz_gate((2 * theta * np.abs(cofactor.imag) * -1).real)
                    q_circuit.append(Rzgate, [control_qubit])
            #                     q_circuit.rz((2 * theta *  np.abs(cofactor.imag) *-1).real, q_register[control_qubit])
            else:
                raise ValueError('PauliWord needs complex part to exponentiate')

    # entangle
    for index, qNo in enumerate(qubitNos[::-1]):
        qNo = int(qNo)
        if min_qubit < qNo:
            next_qubit = int(qubitNos[::-1][index + 1])
            q_circuit.cx(q_register[next_qubit], q_register[qNo])

        # undo basis change
    for index, qNo in enumerate(qubitNos):
        Pstr = PauliStrs[index]
        qNo = int(qNo)
        if Pstr == 'X':
            q_circuit.h(q_register[qNo])
        elif Pstr == 'Y':
            q_circuit.rx((-np.pi / 2), q_register[qNo])
        elif Pstr == 'Z':
            continue

    return q_circuit

def change_basis_for_Z_measure(PauliWord, q_register, q_circuit):
    q_circuit = q_circuit.copy()

    qubitNos, PauliStrs = zip(*list(*PauliWord.terms.keys()))

    # change basis
    for index, qNo in enumerate(qubitNos):
        qNo = int(qNo)
        Pstr = PauliStrs[index]
        if Pstr == 'X':
            q_circuit.h(q_register[qNo])

        elif Pstr == 'Y':
            q_circuit.rx((+np.pi / 2), q_register[qNo])

        elif Pstr == 'Z':
            continue

        else:
            raise ValueError('Not a PauliWord')

    return q_circuit

def arb_state_initalize_circuit(state_to_prepare, q_register, q_circuit):
    q_circuit=q_circuit.copy()
    state_to_prepare = np.asarray(state_to_prepare)
    q_circuit.initialize(state_to_prepare.tolist(), q_register)
    return q_circuit

def Get_Q_circ_to_build_state(arb_state,q_reg, qcirc, check_state=False):
    # https://qiskit.org/documentation/_modules/qiskit/extensions/quantum_initializer/initializer.html

    # assumes logical zero input state.
    # gives quantum circuit to prepare state (use decompose to get standard gates)
    # the qiskit.quantum_info.Operator function can be used to get the unitary matrix of the quantum circuit!

    qcirc = arb_state_initalize_circuit(arb_state, q_reg, qcirc)
    qcirc = qcirc.decompose()

    # need to remove reset part of circuit
    new_data = []
    for index, tup in enumerate(qcirc.data):
        op_type, _, _ = tup
        if isinstance(op_type, qiskit.circuit.reset.Reset):
            continue
        else:
            new_data.append(tup)
    qcirc.data = new_data

    if check_state:
        backend = Aer.get_backend('statevector_simulator')
        job = execute(qcirc, backend)
        qc_state = job.result().get_statevector(qcirc)

        if not np.allclose(qc_state, arb_state):
            raise ValueError('Incorrect state being prepared')

    return qcirc


from qiskit.circuit.library.standard_gates import XGate, YGate, ZGate
def IBM_PauliWord(PauliOp, N_qubits, draw=False, reverse=False):
    qubitNos, PauliStrs = zip(*list(*PauliOp.terms.keys()))
    q_register = QuantumRegister(N_qubits)
    q_circuit = QuantumCircuit(q_register)

    for qNo in range(N_qubits):
        if qNo in qubitNos:
            index = qubitNos.index(qNo)
            Pstr = PauliStrs[index]
            if Pstr == 'X':
                q_circuit.x(q_register[qNo])
            elif Pstr == 'Y':
                q_circuit.y(q_register[qNo])
            elif Pstr == 'Z':
                q_circuit.z(q_register[qNo])
            else:
                raise ValueError('Not a Pauli {}'.format(Pstr))
        else:
            q_circuit.i(q_register[qNo])

    if reverse:
        q_circuit = q_circuit.reverse_bits()
        if draw:
            print(q_circuit.draw())
        return Operator(q_circuit).data
    else:
        if draw:
            print(q_circuit.draw())
        return Operator(q_circuit).data


def Vector_defined_Ansatz(n_qubits, ground_state_vector, check_ansatz_state=False, decompose_fully=False):
    q_reg = QuantumRegister(n_qubits)
    qcirc = QuantumCircuit(q_reg)

    ansatz_circ = Get_Q_circ_to_build_state(ground_state_vector,
                                                    q_reg,
                                                    qcirc,
                                                    check_state=check_ansatz_state)

    ansatz_circ = ansatz_circ.reverse_bits()

    if decompose_fully:
        ansatz_circ = Total_decompose_qiskit(ansatz_circ)

    return ansatz_circ, q_reg

def Build_Standard_VQE_circuits(QubitHamiltonian, Ansatz_circuit, q_reg):



    circuit_list=[]
    for qubitOp in tqdm(QubitHamiltonian, ascii=True, desc='Getting_standard_VQE_circuits'):
        for PauliWord, const in qubitOp.terms.items():
            if PauliWord:
                full_circuit = change_basis_for_Z_measure(qubitOp,
                                                          q_reg,
                                                          Ansatz_circuit,
                                                          )
                full_circuit.measure_all()
                circuit_list.append({'circuit': full_circuit, 'coeff': const, 'qubitOp': qubitOp})
            else:
                I_term = const
                # circuit_list.append({'circuit': None, 'coeff': const})
    return circuit_list, I_term

def Build_Standard_VQE_circuits_MEMORY_EFF(QubitHamiltonian, Ansatz_circuit, q_reg):

    circuit_list=[]
    for qubitOp in tqdm(QubitHamiltonian, ascii=True, desc='Getting_standard_VQE_circuits'):
        for PauliWord, const in qubitOp.terms.items():
            if PauliWord:
                full_circuit = change_basis_for_Z_measure(qubitOp,
                                                          q_reg,
                                                          Ansatz_circuit,
                                                          )
                full_circuit.measure_all()
                qasm_circuit = full_circuit.qasm()
                del full_circuit
                circuit_list.append({'circuit': qasm_circuit, 'coeff': const, 'qubitOp': qubitOp})
            else:
                I_term = const
                # circuit_list.append({'circuit': None, 'coeff': const})
    return circuit_list, I_term

def standard_VQE_lin_alg(QubitHamiltonian, ground_state_vector, n_qubits, check_ansatz_state=False):

    q_reg = QuantumRegister(n_qubits)
    qcirc = QuantumCircuit(q_reg)

    perfect_ansatz_circ = Get_Q_circ_to_build_state(ground_state_vector,
                                                    q_reg,
                                                    qcirc,
                                                    check_state=check_ansatz_state)
    perfect_ansatz_circ = perfect_ansatz_circ.reverse_bits()

    backend = Aer.get_backend('statevector_simulator')
    job = execute(perfect_ansatz_circ, backend)
    ANSATZ_STATE = job.result().get_statevector(perfect_ansatz_circ)
    ANSATZ_bra = ANSATZ_STATE.conj().T

    E_list=[]
    for qubitOp in tqdm(QubitHamiltonian, ascii=True, desc='performing_standard_VQE'):
        for PauliWord, const in qubitOp.terms.items():
            if PauliWord:
                Pauli_matrix = IBM_PauliWord(qubitOp, n_qubits, draw=False, reverse=False)

                exp_val = np.dot(ANSATZ_bra, Pauli_matrix.dot(ANSATZ_STATE))
                E_list.append(exp_val * const)
            else:
                E_list.append(const)
    return sum(E_list)


#### sequence of rotations
from quchem.Unitary_partitioning_Seq_Rot import *
def Build_reduction_circuit_seq_rot_IBM(anti_commuting_set, S_index, q_register, n_qubits, check_reduction=False):
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

    seq_R_circuit = QuantumCircuit(q_register)
    for X_sk_Op, theta_sk in X_sk_theta_sk_list:
        pauliword_X_sk = list(X_sk_Op.terms.keys())[0]
        const_X_sk = list(X_sk_Op.terms.values())[0]

        seq_R_circuit = exp_PauliWord_circuit_IBM(QubitOperator(pauliword_X_sk, -1j), theta_sk / 2 * const_X_sk,
                                                  q_register, seq_R_circuit)

    if check_reduction:

        H_S = QubitOperator()
        for op in full_normalised_set['PauliWords']:
            H_S += op
        H_S_matrix = get_sparse_operator(H_S, n_qubits=n_qubits)

        Ps_mat = get_sparse_operator(Ps, n_qubits=n_qubits)
        R_S_matrix = Operator(seq_R_circuit.reverse_bits()).data

        reduction_mat = R_S_matrix.dot(H_S_matrix.dot(R_S_matrix.conj().transpose()))

        if not (np.allclose(Ps_mat.todense(), reduction_mat)):
            print('reduction circuit incorrect...   𝑅s 𝐻s 𝑅s† != 𝑃s')
    return seq_R_circuit, Ps, gamma_l

def Get_Seq_Rot_Unitary_Part_circuits(anti_commuting_sets, Ansatz_circuit, q_reg, n_qubits, S_index_dict=None,
                                      rotation_reduction_check=False):


    if S_index_dict is None:
        S_index_dict = {key: 0 for key in anti_commuting_sets}


    circuit_list = []
    for set_key in tqdm(list(anti_commuting_sets.keys()), ascii=True, desc='Getting seq_rot VQE circuits'):
        anti_set_list = anti_commuting_sets[set_key]
        if len(anti_set_list) > 1:
            R_sl_circuit, Ps, gamma_l = Build_reduction_circuit_seq_rot_IBM(
                                                                            anti_set_list,
                                                                            S_index_dict[set_key],
                                                                            q_reg,
                                                                            n_qubits,
                                                                            check_reduction=rotation_reduction_check)

            combined_circuits = Ansatz_circuit.combine(R_sl_circuit)

            full_circuit = change_basis_for_Z_measure(Ps,
                                                      q_reg,
                                                      combined_circuits)
            full_circuit.measure_all()
            circuit_list.append({'circuit': full_circuit, 'gamma_l': gamma_l, 'Ps': Ps})
        else:
            qubitOp = anti_set_list[0]
            for PauliWord, const in qubitOp.terms.items():
                if PauliWord:
                    full_circuit = change_basis_for_Z_measure(qubitOp,
                                                              q_reg,
                                                              Ansatz_circuit)
                    full_circuit.measure_all()
                    circuit_list.append({'circuit': full_circuit, 'coeff': const, 'qubitOp': qubitOp})
                else:
                    I_term = const
                    # circuit_list.append({'circuit': None, 'coeff': const})
    return circuit_list, I_term

def Get_Seq_Rot_Unitary_Part_circuits_MEMORY_EFF(anti_commuting_sets, Ansatz_circuit, q_reg, n_qubits, S_index_dict=None,
                                      rotation_reduction_check=False):


    if S_index_dict is None:
        S_index_dict = {key: 0 for key in anti_commuting_sets}


    circuit_list = []
    for set_key in tqdm(list(anti_commuting_sets.keys()), ascii=True, desc='Getting seq_rot VQE circuits'):
        anti_set_list = anti_commuting_sets[set_key]
        if len(anti_set_list) > 1:
            R_sl_circuit, Ps, gamma_l = Build_reduction_circuit_seq_rot_IBM(
                                                                            anti_set_list,
                                                                            S_index_dict[set_key],
                                                                            q_reg,
                                                                            n_qubits,
                                                                            check_reduction=rotation_reduction_check)

            GATES = ['u1', 'u2', 'u3', 'cx']
            R_sl_circuit = transpile(R_sl_circuit,
                                     backend=None,
                                     basis_gates=GATES)

            combined_circuits = Ansatz_circuit.combine(R_sl_circuit)

            full_circuit = change_basis_for_Z_measure(Ps,
                                                      q_reg,
                                                      combined_circuits)
            full_circuit.measure_all()

            qasm_circuit = full_circuit.qasm()
            del full_circuit

            circuit_list.append({'circuit': qasm_circuit, 'gamma_l': gamma_l, 'Ps': Ps})
        else:
            qubitOp = anti_set_list[0]
            for PauliWord, const in qubitOp.terms.items():
                if PauliWord:
                    full_circuit = change_basis_for_Z_measure(qubitOp,
                                                              q_reg,
                                                              Ansatz_circuit)
                    full_circuit.measure_all()

                    qasm_circuit = full_circuit.qasm()
                    del full_circuit

                    circuit_list.append({'circuit': qasm_circuit, 'coeff': const, 'qubitOp': qubitOp})
                else:
                    I_term = const
                    # circuit_list.append({'circuit': None, 'coeff': const})
    return circuit_list, I_term

def Seq_Rot_VQE_lin_alg(anti_commuting_sets, ground_state_vector, n_qubits, S_index_dict=None, rotation_reduction_check=False,
                                        check_ansatz_state=False):

    q_reg = QuantumRegister(n_qubits)
    qcirc = QuantumCircuit(q_reg)

    perfect_ansatz_circ = Get_Q_circ_to_build_state(ground_state_vector,
                                                    q_reg,
                                                    qcirc,
                                                    check_state=check_ansatz_state)

    perfect_ansatz_circ = perfect_ansatz_circ.reverse_bits()  # reverse order here!

    backend = Aer.get_backend('statevector_simulator')
    job = execute(perfect_ansatz_circ, backend)
    ANSATZ_STATE = job.result().get_statevector(perfect_ansatz_circ)
    ANSATZ_bra = ANSATZ_STATE.conj().T


    if S_index_dict is None:
        S_index_dict = {key: 0 for key in anti_commuting_sets}


    E_list=[]
    for set_key in tqdm(list(anti_commuting_sets.keys()), ascii=True, desc='Performing seq_rot VQE lin alg'):
        anti_set_list = anti_commuting_sets[set_key]
        if len(anti_set_list) > 1:
            R_sl_circuit, Ps, gamma_l = Build_reduction_circuit_seq_rot_IBM(
                anti_set_list,
                S_index_dict[set_key],
                q_reg,
                n_qubits,
                check_reduction=rotation_reduction_check)

            R_sl = Operator(R_sl_circuit).data  # may need to reverse bits!
            Pauli_matrix = IBM_PauliWord(Ps, n_qubits, draw=False, reverse=False)

            post_Rsl_state = R_sl.dot(ANSATZ_STATE)

            exp_val = np.dot(post_Rsl_state.conj().T, Pauli_matrix.dot(post_Rsl_state))
            E_list.append(exp_val * gamma_l)
        else:
            qubitOp = anti_set_list[0]
            for PauliWord, const in qubitOp.terms.items():
                if PauliWord:
                    Pauli_matrix = IBM_PauliWord(qubitOp, n_qubits, draw=False, reverse=False)
                    exp_val = np.dot(ANSATZ_bra, Pauli_matrix.dot(ANSATZ_STATE))
                    E_list.append(exp_val * const)
                else:
                    E_list.append(const)
    return sum(E_list)


#### Linear Combination of unitaries
from quchem.Unitary_partitioning_LCU_method import *

def phase_Pauli_gate(Pstr, cofactor):
    if Pstr == 'X':
        unitary_matrix = cofactor * np.array([[0, 1], [1, 0]], dtype=complex)
    elif Pstr == 'Y':
        unitary_matrix = cofactor * np.array([[0, -1j], [1j, 0]], dtype=complex)
    elif Pstr == 'Z':
        unitary_matrix = cofactor * np.array([[1, 0], [0, -1]], dtype=complex)
    else:
        raise ValueError('P_str is not a Pauli')

    return UnitaryGate(unitary_matrix, label='{}*{}'.format(cofactor, Pstr))

def control_P_IBM(PauliOp, phase_correction, control_index, q_circuit, n_qubits, n_ancilla, list_measured_qubits=None):
    q_circuit = q_circuit.copy()
    qubitNos, PauliStrs = zip(*list(*PauliOp.terms.keys()))

    control_indices_list = list(range(n_qubits, n_qubits + n_ancilla))

    if list_measured_qubits is None:
        qubit_to_put_phase_on = 0
    else:
        qubit_to_put_phase_on = list(set(qubitNos) & set(list_measured_qubits))[0]

    for index, qNo in enumerate(qubitNos):
        qNo = int(qNo)
        Pstr = PauliStrs[index]

        if qNo == qubit_to_put_phase_on:
            phase_P = phase_Pauli_gate(Pstr, phase_correction).control(n_ancilla)
            phase_P.num_ctrl_qubits = n_ancilla
            phase_P.ctrl_state = control_index
            #             q_circuit.append(phase_P, [*[i for i in range(0, n_ancilla)],qNo])
            q_circuit.append(phase_P, [*control_indices_list, qNo])
        else:
            if Pstr == 'X':
                X_gate = XGate().control(n_ancilla)
                X_gate.ctrl_state = control_index
                q_circuit.append(X_gate, [*control_indices_list, qNo])
            #                 q_circuit.append(X_gate, [*[i for i in range(0, n_ancilla)],qNo])
            elif Pstr == 'Y':
                Y_gate = YGate().control(n_ancilla)
                Y_gate.ctrl_state = control_index
                q_circuit.append(Y_gate, [*control_indices_list, qNo])
            #                 q_circuit.append(Y_gate, [*[i for i in range(0, n_ancilla)],qNo])
            elif Pstr == 'Z':
                Z_gate = ZGate().control(n_ancilla)
                Z_gate.ctrl_state = control_index
                q_circuit.append(Z_gate, [*control_indices_list, qNo])
    #                 q_circuit.append(Z_gate, [*[i for i in range(0, n_ancilla)],qNo])

    return q_circuit

def Get_post_selection_counts_LCU(list_of_measurements, N_ancilla):
    # checks all zero state on ancilla line
    # ancilla must be first part of measurment string!
    # requires circuit simulation memory=True

    new_counts = {}
    for binary_result_str in list_of_measurements:
        ancilla_state = int(binary_result_str[:N_ancilla], 2)
        if ancilla_state == 0:
            post_select_m_binary = binary_result_str[N_ancilla:]
            if post_select_m_binary in new_counts.keys():
                new_counts[post_select_m_binary] += 1
            else:
                new_counts[post_select_m_binary] = 1

        else:
            continue

    return new_counts

def Get_post_selection_counts_DICT_LCU(count_dict, N_ancilla):
    # checks all zero state on ancilla line
    # ancilla must be first part of measurment string!
    # requires circuit simulation memory=True

    new_counts = {}
    for binary_result_str in count_dict:
        ancilla_state = int(binary_result_str[:N_ancilla], 2)
        if ancilla_state == 0:
            post_select_m_binary = binary_result_str[N_ancilla:]
            new_counts[post_select_m_binary] = count_dict[binary_result_str]
        else:
            continue

    return new_counts


def Get_LCU_Unitary_Part_circuits(anti_commuting_sets, Ansatz_circuit, q_reg, n_qubits,
                                  N_index_dict=None):

    if N_index_dict is None:
        N_index_dict = {key: 0 for key in anti_commuting_sets}


    circuit_list = []
    for set_key in tqdm(list(anti_commuting_sets.keys()), ascii=True, desc='Getting LCU VQE circuits'):
        anti_set_list = anti_commuting_sets[set_key]
        if len(anti_set_list) > 1:
            R_uncorrected, Pn, gamma_l = Get_R_op_list(anti_set_list, N_index_dict[set_key])
            R_corrected_Op_list, R_corr_list, ancilla_amplitudes, l1 = absorb_complex_phases(R_uncorrected)

            N_ancilla = int(np.ceil(np.log2(len(ancilla_amplitudes))))
            if len(ancilla_amplitudes) != 2 ** N_ancilla:
                n_missing = int(2 ** N_ancilla - len(ancilla_amplitudes))
                missing_terms = [0 for _ in range(n_missing)]
                ancilla_amplitudes = [*ancilla_amplitudes, *missing_terms]

            q_reg_ancilla = QuantumRegister(N_ancilla)
            q_circ_ancilla = QuantumCircuit(q_reg_ancilla)
            G_circuit = Get_Q_circ_to_build_state(ancilla_amplitudes, q_reg_ancilla, q_circ_ancilla)
            G_inverse = G_circuit.inverse()

            # combine ancilla and system
            combined_circuits = Ansatz_circuit.combine(G_circuit)

            # find qubits that are measured!
            Pn_qubitNos, _ = zip(*list(*Pn.terms.keys()))

            for control_index, op in enumerate(R_corrected_Op_list):
                phase_corr = R_corr_list[control_index]
                for PauliW, Const in op.terms.items():
                    if PauliW:
                        combined_circuits = control_P_IBM(op,
                                                          phase_corr,
                                                          control_index,
                                                          combined_circuits,
                                                          n_qubits,
                                                          N_ancilla,
                                                          list_measured_qubits=Pn_qubitNos)
                    else:
                        continue

            # G dag
            combined_circuits = combined_circuits.combine(G_inverse)

            full_circuit = change_basis_for_Z_measure(Pn,
                                                      q_reg,
                                                      combined_circuits)
            full_circuit.measure_all()

            circuit_list.append({'circuit': full_circuit, 'gamma_l': gamma_l, 'Pn': Pn, 'N_ancilla': N_ancilla})
        else:
            qubitOp = anti_set_list[0]
            for PauliWord, const in qubitOp.terms.items():
                if PauliWord:
                    full_circuit = change_basis_for_Z_measure(qubitOp,
                                                              q_reg,
                                                              Ansatz_circuit)
                    full_circuit.measure_all()
                    circuit_list.append({'circuit': full_circuit, 'coeff': const, 'qubitOp': qubitOp})
                else:
                    I_term = const
                    # circuit_list.append({'circuit': None, 'coeff': const})
    return circuit_list, I_term


from qiskit.compiler import transpile
def Get_LCU_Unitary_Part_circuits_MEMORY_EFF(anti_commuting_sets, Ansatz_circuit, q_reg, n_qubits,
                                             N_index_dict=None):
    if N_index_dict is None:
        N_index_dict = {key: 0 for key in anti_commuting_sets}

    circuit_list = []
    for set_key in tqdm(list(anti_commuting_sets.keys()), ascii=True, desc='Getting LCU VQE circuits'):
        anti_set_list = anti_commuting_sets[set_key]
        if len(anti_set_list) > 1:
            R_uncorrected, Pn, gamma_l = Get_R_op_list(anti_set_list, N_index_dict[set_key])
            R_corrected_Op_list, R_corr_list, ancilla_amplitudes, l1 = absorb_complex_phases(R_uncorrected)

            N_ancilla = int(np.ceil(np.log2(len(ancilla_amplitudes))))
            if len(ancilla_amplitudes) != 2 ** N_ancilla:
                n_missing = int(2 ** N_ancilla - len(ancilla_amplitudes))
                missing_terms = [0 for _ in range(n_missing)]
                ancilla_amplitudes = [*ancilla_amplitudes, *missing_terms]

            q_reg_ancilla = QuantumRegister(N_ancilla)
            q_circ_ancilla = QuantumCircuit(q_reg_ancilla)

            G_circuit = Get_Q_circ_to_build_state(ancilla_amplitudes, q_reg_ancilla, q_circ_ancilla)
            G_circuit = Total_decompose_qiskit(G_circuit)

            G_inverse = G_circuit.inverse()

            # combine ancilla and system
            combined_circuits = Ansatz_circuit.combine(G_circuit)

            # find qubits that are measured!
            Pn_qubitNos, _ = zip(*list(*Pn.terms.keys()))

            for control_index, op in enumerate(R_corrected_Op_list):
                phase_corr = R_corr_list[control_index]
                for PauliW, Const in op.terms.items():
                    if PauliW:
                        combined_circuits = control_P_IBM(op,
                                                          phase_corr,
                                                          control_index,
                                                          combined_circuits,
                                                          n_qubits,
                                                          N_ancilla,
                                                          list_measured_qubits=Pn_qubitNos)
                    else:
                        continue

            # G dag
            combined_circuits = combined_circuits.combine(G_inverse)

            # decomposed
            combined_circuits = Total_decompose_qiskit(combined_circuits)

            full_circuit = change_basis_for_Z_measure(Pn,
                                                      q_reg,
                                                      combined_circuits)
            full_circuit.measure_all()

            GATES = ['u1', 'u2', 'u3', 'cx']
            full_circuit = transpile(full_circuit,
                                     backend=None,
                                     basis_gates=GATES)

            qasm_circuit = full_circuit.qasm()
            del full_circuit

            circuit_list.append({'circuit': qasm_circuit, 'gamma_l': gamma_l, 'Pn': Pn, 'N_ancilla': N_ancilla})
        else:
            qubitOp = anti_set_list[0]
            for PauliWord, const in qubitOp.terms.items():
                if PauliWord:
                    full_circuit = change_basis_for_Z_measure(qubitOp,
                                                              q_reg,
                                                              Ansatz_circuit)
                    full_circuit.measure_all()

                    qasm_circuit = full_circuit.qasm()
                    del full_circuit

                    circuit_list.append({'circuit': qasm_circuit, 'coeff': const, 'qubitOp': qubitOp})
                else:
                    I_term = const
                    # circuit_list.append({'circuit': None, 'coeff': const})
    return circuit_list, I_term

def POVM_LCU(n_system_q, n_ancilla_q, system_ancilla_output_ket):
    # state_vector_simulator the state is given as (ancilla X_kron system)

    full_density_matrix = np.outer(system_ancilla_output_ket, system_ancilla_output_ket)

    I_system_operator = np.eye((2 ** n_system_q))

    ancilla_0_state = np.eye(2 ** n_ancilla_q)[0, :]
    ancilla_0_projector = np.outer(ancilla_0_state, ancilla_0_state)

    POVM_0_ancilla = np.kron(ancilla_0_projector, I_system_operator)
    Kraus_Op_0 = POVM_0_ancilla.copy()

    term = Kraus_Op_0.dot(full_density_matrix.dot(Kraus_Op_0.transpose().conj()))
    projected_density_matrix = term / np.trace(term)  # projected into correct space using POVM ancilla measurement!

    #     ## Next get partial density matrix over system qubits # aka partial trace!
    #     # https://scicomp.stackexchange.com/questions/27496/calculating-partial-trace-of-array-in-numpy
    #     # reshape to do the partial trace easily using np.einsum

    #     reshaped_dm = projected_density_matrix.reshape([2 ** n_system_q, 2 ** n_ancilla_q,
    #                                                     2 ** n_system_q, 2 ** n_ancilla_q])
    #     reduced_dm = np.einsum('jiki->jk', reshaped_dm)

    # p_a = sum_{b} (I_{a}*<b|) p_{ab} (I_{a}*|b>)
    basis_ancilla = np.eye((2 ** n_ancilla_q))
    reduced_dm = np.zeros((2 ** n_system_q, 2 ** n_system_q), dtype=complex)
    for b in range(basis_ancilla.shape[0]):
        b_ket = basis_ancilla[b, :].reshape([2 ** n_ancilla_q, 1])
        I_a_b_ket = np.kron(b_ket, I_system_operator)
        #         I_a_b_ket = np.kron(I_system_operator, b_ket)
        I_a_b_bra = I_a_b_ket.transpose().conj()

        term = I_a_b_bra.dot(projected_density_matrix.dot(I_a_b_ket))
        reduced_dm += term

    return reduced_dm

def LCU_VQE_lin_alg(anti_commuting_sets, ground_state_vector, n_qubits, N_index_dict=None,
                                        check_ansatz_state=False):

    q_reg = QuantumRegister(n_qubits)
    qcirc = QuantumCircuit(q_reg)

    perfect_ansatz_circ = Get_Q_circ_to_build_state(ground_state_vector,
                                                    q_reg,
                                                    qcirc,
                                                    check_state=check_ansatz_state)

    perfect_ansatz_circ = perfect_ansatz_circ.reverse_bits()  # reverse order here!

    backend = Aer.get_backend('statevector_simulator')
    job = execute(perfect_ansatz_circ, backend)
    ANSATZ_STATE = job.result().get_statevector(perfect_ansatz_circ)
    ANSATZ_bra = ANSATZ_STATE.conj().T


    if N_index_dict is None:
        N_index_dict = {key: 0 for key in anti_commuting_sets}


    E_list=[]
    for set_key in tqdm(list(anti_commuting_sets.keys()), ascii=True, desc='Performing LCU VQE lin alg'):
        anti_set_list = anti_commuting_sets[set_key]
        if len(anti_set_list) > 1:
            R_uncorrected, Pn, gamma_l = Get_R_op_list(anti_set_list, N_index_dict[set_key])
            R_corrected_Op_list, R_corr_list, ancilla_amplitudes, l1 = absorb_complex_phases(R_uncorrected)

            N_ancilla = int(np.ceil(np.log2(len(ancilla_amplitudes))))
            if len(ancilla_amplitudes) != 2 ** N_ancilla:
                n_missing = int(2 ** N_ancilla - len(ancilla_amplitudes))
                missing_terms = [0 for _ in range(n_missing)]
                ancilla_amplitudes = [*ancilla_amplitudes, *missing_terms]

            q_reg_ancilla = QuantumRegister(N_ancilla)
            q_circ_ancilla = QuantumCircuit(q_reg_ancilla)
            G_circuit = Get_Q_circ_to_build_state(ancilla_amplitudes, q_reg_ancilla, q_circ_ancilla)
            G_inverse = G_circuit.inverse()

            # combine ancilla and system
            combined_circuits = perfect_ansatz_circ.combine(G_circuit)

            # find qubits that are measured!
            Pn_qubitNos, _ = zip(*list(*Pn.terms.keys()))

            for control_index, op in enumerate(R_corrected_Op_list):
                phase_corr = R_corr_list[control_index]
                for PauliW, Const in op.terms.items():
                    if PauliW:
                        combined_circuits = control_P_IBM(op,
                                                          phase_corr,
                                                          control_index,
                                                          combined_circuits,
                                                          n_qubits,
                                                          N_ancilla,
                                                          list_measured_qubits=Pn_qubitNos)
                    else:
                        continue

            # G dag
            combined_circuits = combined_circuits.combine(G_inverse)
            job = execute(combined_circuits, backend)
            ANSATZ_and_ANCILLA = job.result().get_statevector(combined_circuits)

            partial_density_matrix = POVM_LCU(n_qubits, N_ancilla, ANSATZ_and_ANCILLA)

            Pn_system_only = IBM_PauliWord(Pn, n_qubits, draw=False, reverse=False)

            energy = np.trace(partial_density_matrix.dot(Pn_system_only))

            E_list.append(energy * gamma_l)
        else:
            qubitOp = anti_set_list[0]
            for PauliWord, const in qubitOp.terms.items():
                if PauliWord:
                    Pauli_matrix = IBM_PauliWord(qubitOp, n_qubits, draw=False, reverse=False)
                    exp_val = np.dot(ANSATZ_bra, Pauli_matrix.dot(ANSATZ_STATE))
                    E_list.append(exp_val * const)
                else:
                    E_list.append(const)
    return sum(E_list)



def Save_exp_inputs(filename, Hamiltonian, anti_commuting_sets, geometry, basis_set, transformation,
                    Graph_colouring_strategy, fci_energy,
                    standard_VQE_circuits, standard_I_term,
                    Seq_Rot_VQE_circuits, Seq_Rot_I_term,
                    LCU_VQE_circuits, LCU_I_term,
                    ground_state_vector,
                    n_system_qubits,
                    S_index_dict=None,
                    N_index_dict=None):
    output={}
    output['Hamiltonian'] = Hamiltonian
    output['anti_commuting_sets'] = anti_commuting_sets
    output['geometry'] = geometry
    output['basis_set'] = basis_set
    output['transformation'] = transformation
    output['Graph_colouring_strategy'] = Graph_colouring_strategy
    output['fci_energy'] = fci_energy

    output['standard_VQE_circuits'] = standard_VQE_circuits
    output['standard_I_term'] = standard_I_term

    output['Seq_Rot_VQE_circuits'] = Seq_Rot_VQE_circuits
    output['S_indices_dict'] = S_index_dict
    output['Seq_Rot_I_term'] = Seq_Rot_I_term

    output['LCU_VQE_circuits'] = LCU_VQE_circuits
    output['LCU_I_term'] = LCU_I_term
    output['N_index_dict'] = N_index_dict

    output['n_system_qubits'] = n_system_qubits
    output['ground_state'] = ground_state_vector

    time = datetime.datetime.now().strftime('%Y%b%d-%H%M%S%f')
    F_name = '{}_time={}'.format(filename, time)

    base_dir = os.getcwd()
    input_dir = os.path.join(base_dir, 'Input_data')

    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

    filepath = os.path.join(input_dir, F_name)

    with open(filepath + '.pickle', 'wb') as fhandle:
        pickle.dump(output, fhandle, protocol=pickle.HIGHEST_PROTOCOL)

    print('experiment data saved here: {}'.format(filepath))

def calc_exp_pauliword(count_dict, PauliWord):
    # takes correct part of bitstring when all lines measured

    qubitNos, PauliStrs = zip(*list(*PauliWord.terms.keys()))
    n_zeros = 0
    n_ones = 0

    for bitstring in count_dict:
        measure_term = np.take([int(bit) for bit in bitstring[::-1]], qubitNos) #reverse order here!

        parity_m_term = sum(measure_term) % 2

        if parity_m_term == 0:
            n_zeros += count_dict[bitstring]
        elif parity_m_term == 1:
            n_ones += count_dict[bitstring]
        else:
            raise ValueError('state {} not allowed'.format(measure_term))

    expectation_value = (n_zeros - n_ones) / (n_zeros + n_ones)

    return expectation_value

def Get_post_selection_counts_LCU(list_of_measurements, N_ancilla):
    # checks all zero state on ancilla line

    new_counts = {}

    if N_ancilla == 0:
        for binary_result_str in list_of_measurements:
            if binary_result_str in new_counts.keys():
                new_counts[binary_result_str] += 1
            else:
                new_counts[binary_result_str] = 1
    else:
        for binary_result_str in list_of_measurements:
            ancilla_state = int(binary_result_str[:N_ancilla], 2)
            if ancilla_state == 0:
                post_select_m_binary = binary_result_str[N_ancilla:]
                if post_select_m_binary in new_counts.keys():
                    new_counts[post_select_m_binary] += 1
                else:
                    new_counts[post_select_m_binary] = 1
            else:
                continue

    return new_counts

from scipy.linalg import svd
def Gram_Schimdt(arb_state):
    # Get an orthonormal basis from a single vector (defines first column of output!)
    # Returns unitary matrix to generate arb state from all zero state
    # not only seems to be working for REAL AMPLITUDES!

    # WORKING
    # https://stackoverflow.com/questions/12327479/how-to-build-a-ortoghonal-basis-from-a-vector

    arb_state = np.ravel(arb_state)

    if not np.isclose(sum(np.abs(arb_state ** 2)), 1):
        raise ValueError('state not normalised')

    n_qubits = len(arb_state)
    V = np.eye(n_qubits, dtype=complex)
    V[:, 0] = arb_state

    U = np.zeros(V.shape, dtype=complex)
    U[:, 0] = V[:, 0]
    for i in range(1, V.shape[0]):
        U[:, i] = V[:, i]
        for j in range(i):
            U[:, i] = U[:, i] - (U[:, j].T.dot(U[:, i]) / (U[:, j].T.dot(U[:, j]))) * U[:, j]

    Unitary_matrx, s, Vh = svd(U)

    # correct the sign
    if not np.allclose(Unitary_matrx[:, 0], arb_state):
        Unitary_matrx[:, 0] = Unitary_matrx[:, 0] * -1

    if not np.allclose(Unitary_matrx[:, 0], arb_state):
        raise ValueError('incorrect state generated')
    return Unitary_matrx