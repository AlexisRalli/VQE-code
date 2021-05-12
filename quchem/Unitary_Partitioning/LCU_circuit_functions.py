from openfermion.ops import QubitOperator
from openfermion import qubit_operator_sparse
import numpy as np
import cirq

from quchem.Qcircuit.misc_quantum_circuit_functions import Get_state_as_str
from quchem.Qcircuit.Circuit_functions_to_create_arb_state import intialization_circuit
from quchem.Qcircuit.Hamiltonian_term_measurement_functions import Change_PauliWord_measurement_to_Z_basis


def absorb_complex_phases(R_linear_comb_list):
    """
    Function that absorbs phase of list of linear combination of unitary operators and divides performs the requried
    l1 normalization.

    ##
    LCU_Op = ∑_{i=0} 𝛼_{j} U_{j}

    used to find:
    l1_norm = ∑_{i=0} |𝛼_{𝑗}|

    all 𝛼_{j} made positive and phase correction held in R_linear_comb_correction_values list!
    ancilla_amplitudes = ∑_{i=0} (𝛼_{𝑗} / l1_norm)^{0.5} |j>
    new_LCU_op is amde as before, but with real positive amps = LCU_Op = ∑_{i=0} |𝛼_{i}^{new}| U_{j}^{new}

    Args:
        R_linear_comb_list (list): list of qubit operators

    Returns:
        R_linear_comb_phase_values (list): list of phase corrections, which allows all terms in LCU sum to be positive.
        R_op_list_real_positive_amps (QubitOperator): list of qubit operators of LCU_op where all amps real and positive
        ancilla_amplitudes: ancillar amplitudes for control U (LCU method) (note l1_normalization)
        l1_norm (float): l1 norm

         """
    R_op_list_real_positive_amps = []
    R_linear_comb_phase_values = []
    ancilla_amplitudes = []

    l1_norm = sum([np.absolute(const) for qubitOp in R_linear_comb_list for PauliWord, const in qubitOp.terms.items()])

    for qubitOp in R_linear_comb_list:
        pauliword, const = tuple(*qubitOp.terms.items())
        ancilla_amplitudes.append(np.sqrt(np.absolute(const) / l1_norm))

        if (isinstance(const, complex)) and (const.imag < 0):
            R_op_list_real_positive_amps.append(QubitOperator(pauliword, np.absolute(const)))
            R_linear_comb_phase_values.append(-1j)
        elif (isinstance(const, complex)) and (const.imag != 0):
            R_op_list_real_positive_amps.append(QubitOperator(pauliword, np.absolute(const)))
            R_linear_comb_phase_values.append(1j)
        elif const < 0:
            R_op_list_real_positive_amps.append(QubitOperator(pauliword, np.absolute(const)))
            R_linear_comb_phase_values.append(-1)
        else:
            R_op_list_real_positive_amps.append(QubitOperator(pauliword, np.absolute(const)))
            R_linear_comb_phase_values.append(1)

    if not np.isclose(sum(np.absolute(amp) ** 2 for amp in ancilla_amplitudes), 1):
        raise ValueError('ancilla amplitudes NOT normalised properly')

    # ## can check if requried:
    # if [P_op*R_linear_comb_phase_values[ind] for ind, P_op in enumerate(R_op_list_real_positive_amps)] != R_linear_comb_list:
    #     raise ValueError('R with positive amplitudes not defined correctly')

    return R_op_list_real_positive_amps, R_linear_comb_phase_values, ancilla_amplitudes, l1_norm

class singeQ_Pauligate_phase(cirq.SingleQubitGate):
    """

    Function performs a pauligate multiplied by a phase (1, -1, 1j, -1j).

    Args:
        PauliStr (str): string of Pauli operator to be performed
        correction_value (complex): constant to multiply gate by

    Returns
        A cirq circuit object to be used by cirq.Circuit

    examples:
    
        gate_obj = singeQ_phase_Pauligate('X', -1j)
        circuit = cirq.Circuit(gate_obj.on(cirq.LineQubit(1)))
        print(circuit)

        >> 1: ───(-0-1j) X───
        
        print(
                cirq.Circuit(cirq.decompose(circuit)))
                
        >> 1: ───Z───Y───

    """

    def __init__(self, PauliStr, correction_value):

        self.PauliStr = PauliStr
        self.correction_value = correction_value

    def _decompose_(self, qubits):
        qubit = qubits[0]
        #########
        if self.PauliStr=='Z':
            if self.correction_value.imag==-1:
                # Y X = -1jZ
                yield cirq.X.on(qubit)
                yield cirq.Y.on(qubit)
            elif self.correction_value.imag==1:
                # X Y = 1jZ
                yield cirq.Y.on(qubit)
                yield cirq.X.on(qubit)
                
            elif self.correction_value.real==-1:
                yield cirq.ry(2*np.pi).on(qubit)
                yield cirq.Z.on(qubit)
            elif self.correction_value.real==1:
                yield cirq.Z.on(qubit)
            else:
                raise ValueError(f'phase does have magnitude of 1: {self.correction_value}')
        ###############
        elif self.PauliStr=='Y':
            if self.correction_value.imag==-1:
                # X Z = -1jZ
                yield cirq.Z.on(qubit)
                yield cirq.X.on(qubit)
            elif self.correction_value.imag==1:
                # Z X = 1jZ
                yield cirq.X.on(qubit)
                yield cirq.Z.on(qubit)
            elif self.correction_value.real==-1:
                yield cirq.ry(2*np.pi).on(qubit)
                yield cirq.Y.on(qubit)
            elif self.correction_value.real==1:
                yield cirq.Y.on(qubit)
            else:
                raise ValueError(f'phase does have magnitude of 1: {self.correction_value}')
        
        ######################
        elif self.PauliStr=='X':
            if self.correction_value.imag==-1:
                # Z Y = -1jX
                yield cirq.Y.on(qubit)
                yield cirq.Z.on(qubit)
            elif self.correction_value.imag==1:
                # Y Z = 1j X
                yield cirq.Z.on(qubit)
                yield cirq.Y.on(qubit)
                
            elif self.correction_value.real==-1:
                yield cirq.ry(2*np.pi).on(qubit)
                yield cirq.X.on(qubit)
            elif self.correction_value.real==1:
                yield cirq.X.on(qubit)
            else:
                raise ValueError(f'phase does have magnitude of 1: {self.correction_value}')
                
        ######################
        elif self.PauliStr=='I':
            if self.correction_value.imag==-1:
                # X Z Y = -1j i
                yield cirq.Y.on(qubit)
                yield cirq.Z.on(qubit)
                yield cirq.X.on(qubit)
            elif self.correction_value.imag==1:
                # Z X Y = +1j i
                yield cirq.Y.on(qubit)
                yield cirq.X.on(qubit)
                yield cirq.Z.on(qubit) 
                
            elif self.correction_value.real==-1:
                yield cirq.ry(2*np.pi).on(qubit)
            elif self.correction_value.real==1:
                yield cirq.I.on(qubit)
            else:
                raise ValueError(f'phase does have magnitude of 1: {self.correction_value}')
        else:
            raise ValueError(f'not a Pauli operation: {self.PauliStr}')
        


    def _circuit_diagram_info_(self, args):
        return '{} {}'.format(str(self.correction_value), self.PauliStr)
        
    def num_qubits(self):
        return 1

class Modified_PauliWord_gate(cirq.Gate):
    """
    Class to generate cirq circuit as a gate that performs a modified PauliWord

    Args:
        PauliQubitOp (QubitOperator): Qubit operator to perform as a circuit
        correction_val (complex): constant to multiply qubit operator by (intended to be phase factor for LCU)

    Returns
        A cirq circuit object to be used by cirq.Circuit

    Example:

    PauliQubitOp = QubitOperator('Y0 X1 X2 X3',1)
    Pn = QubitOperator('Z2',1)
    correction_val = -1j
    circuit_obj = Modified_PauliWord_gate(PauliQubitOp, correction_val, Pn).on(*list(cirq.LineQubit.range(4)))
    circuit=cirq.Circuit(circuit_obj)
    
    print(circuit)

    

    >> 

        0: ───Y0───────────
              │
        1: ───X1───────────
              │
        2: ───(-0-1j)*X2───
              │
        3: ───X3───────────

    print(cirq.Circuit(cirq.decompose(circuit)))
    >> 

        0: ───Y───────

        1: ───X───────

        2: ───Y───Z───

        3: ───X───────

    """

    def __init__(self, PauliQubitOp, correction_val, Pn):

        self.PauliQubitOp = PauliQubitOp
        self.correction_val = correction_val

        self.list_of_X_qNos_Pn, list_of_Pn_ops = list(zip(*[Paulistrs for qubitOp in Pn
                                                            for Paulistrs, const in qubitOp.terms.items()][0]))
        # self.sign_index = self.list_of_X_qNos_Pn[0]

        if list(PauliQubitOp.terms.keys())[0]!=():
            PauliQubitOp_Qnumbers, _ = list(zip(*[Paulistrs for qubitOp in PauliQubitOp
                                    for Paulistrs, const in qubitOp.terms.items()][0]))

            self.sign_index = np.intersect1d(self.list_of_X_qNos_Pn, PauliQubitOp_Qnumbers)[0]
        else:
            self.sign_index = self.list_of_X_qNos_Pn[0]

    def __repr__(self):
        return 'LCU_Pauli_Word_Gates'

    def _decompose_(self, qubits):

        if list(self.PauliQubitOp.terms.keys())[0] == ():
            # identity operations
            yield singeQ_Pauligate_phase('I', self.correction_val).on(qubits[self.sign_index])
        else:
            qubitNos_list, P_strs_list = zip(*list(self.PauliQubitOp.terms.keys())[0])

            for index, qNo in enumerate(qubitNos_list):
                P_str = P_strs_list[index]
                if qNo == self.sign_index:
                    yield singeQ_Pauligate_phase(P_str, self.correction_val).on(qubits[qNo])
                else:
                    if P_str == 'Z':
                        yield cirq.Z.on(qubits[qNo])
                    elif P_str == 'Y':
                        yield cirq.Y.on(qubits[qNo])
                    elif P_str == 'X':
                        yield cirq.X.on(qubits[qNo])
                    elif P_str == 'I':
                        yield cirq.I.on(qubits[qNo])
                    else:
                        raise ValueError('Not a Pauli Operation')

    def _circuit_diagram_info_(self, args):

        if list(self.PauliQubitOp.terms.keys())[0] == ():

            string_list = []
            for qubitNo in range(self.num_qubits()):

                if qubitNo == self.sign_index:
                    string_list.append('{}*{}{}'.format(self.correction_val, 'I', self.sign_index))
                else:
                    string_list.append('I')
            return string_list

        else:

            qubitNos_list, P_strs_list = zip(*list(self.PauliQubitOp.terms.keys())[0])

            string_list = []
            for qubitNo in range(self.num_qubits()):
                if qubitNo in qubitNos_list:
                    P_string_index = qubitNos_list.index(qubitNo)
                    if qubitNo == self.sign_index:
                        string_list.append('{}*{}{}'.format(self.correction_val, P_strs_list[P_string_index], qubitNo))
                    else:
                        string_list.append('{}{}'.format(P_strs_list[P_string_index], qubitNo))
                else:
                    string_list.append('I')

            return string_list

    def num_qubits(self):
        if list(self.PauliQubitOp.terms.keys())[0] == ():
            return self.sign_index + 1
        else:
            qubitNos_list, P_strs_list = zip(*list(self.PauliQubitOp.terms.keys())[0])
            return max(qubitNos_list) + 1

class LCU_U_gate(cirq.Gate):
    """
    Function to build cirq Circuit that performs controlled modified pauligate for LCU method

    Args:
        circuit_param_dict (dict): A Dictionary of Tuples (qubit, control_val(int)) value is angle

    Returns
        A cirq circuit object to be used by cirq.Circuit.from_ops to generate arbitrary state

    """

    def __init__(self, No_control_qubits, No_system_qubits, R_corrected_Op_list, R_correction_list, Pn):

        self.No_control_qubits = No_control_qubits
        self.No_system_qubits = No_system_qubits
        self.R_corrected_Op_list = R_corrected_Op_list
        self.R_correction_list = R_correction_list
        self.Pn = Pn

    def _decompose_(self, qubits):

        if self.No_control_qubits!=  np.ceil(np.log2(len(self.R_corrected_Op_list))):
            raise ValueError('error in number of Ancilla qubits')

        for control_state_index, R_qubitOp_corrected in enumerate(self.R_corrected_Op_list):

            if list(R_qubitOp_corrected.terms.keys())[0] != ():

                control_str = Get_state_as_str(self.No_control_qubits, control_state_index)
                control_values = [int(bit) for bit in control_str]

                qubitNos_list, _ = zip(*list(R_qubitOp_corrected.terms.keys())[0])

                qubit_list = cirq.LineQubit.range(self.No_system_qubits, self.No_system_qubits + self.No_control_qubits) \
                             + cirq.LineQubit.range(qubitNos_list[-1] + 1)  # note control qubits first!

                mod_p_word_gate = Modified_PauliWord_gate(R_qubitOp_corrected,
                                                             self.R_correction_list[control_state_index], self.Pn)

                yield mod_p_word_gate.controlled(num_controls=self.No_control_qubits, control_values=control_values).on(
                    *qubit_list)  # *qubit_list
            else:
                #identity circuit
                if self.R_correction_list[control_state_index]!=1: # checks if phase correction is required!
                    control_str = Get_state_as_str(self.No_control_qubits, control_state_index)
                    control_values = [int(bit) for bit in control_str]

                    list_of_X_qNos_Pn, _ = list(
                        zip(*[Paulistrs for qubitOp in self.Pn for Paulistrs, const in qubitOp.terms.items()][0]))
                    No_I_qubit_to_Operate = list_of_X_qNos_Pn[0] + 1

                    qubit_list = cirq.LineQubit.range(self.No_system_qubits, self.No_system_qubits + self.No_control_qubits) \
                                 + cirq.LineQubit.range(No_I_qubit_to_Operate)  # note control qubits first!

                    mod_p_word_gate = Modified_PauliWord_gate(R_qubitOp_corrected,
                                                                 self.R_correction_list[control_state_index], self.Pn)

                    yield mod_p_word_gate.controlled(num_controls=self.No_control_qubits, control_values=control_values).on(
                        *qubit_list)  # *qubit_list

    def _circuit_diagram_info_(self, args):

        string_list = []

        for _ in range(self.No_system_qubits):
            string_list.append('Pauli_Mod_Cirq_LCU')

        for _ in range(self.No_control_qubits):
            string_list.append('control_LCU')

        return string_list

    def num_qubits(self):
        return self.No_control_qubits + self.No_system_qubits

class Measure_system_and_ancilla(cirq.Gate):
    """
    Class to generate cirq circuit that measures PauliWord in Z BASIS AND ancilla line!!!!

    Args:
        PauliQubitOp (QubitOperator): Qubit operator to measure
        N_ancilla_qubits (int): Number of ancilla qubits
        N_system_qubits (int): Number of system qubits / spin orbitals

    Returns
        A cirq circuit object to be used by cirq.Circuit

    example

    N_ancilla_qubits = 2
    N_system_qubits = 10
    OP_to_measure = QubitOperator('X0 Z2 Y3', 0.25j)

    output:

    0: ────M───
           │
    2: ────M───
           │
    3: ────M───
           │
    10: ───M───
           │
    11: ───M───

    """

    def __init__(self, PauliQubitOp, N_ancilla_qubits, N_system_qubits):

        self.PauliQubitOp = PauliQubitOp
        self.N_ancilla_qubits = N_ancilla_qubits
        self.N_system_qubits = N_system_qubits

    def _decompose_(self, qubits):

        qubit_system_list, _ = zip(*list(*self.PauliQubitOp.terms.keys()))
        qubit_ancilla_list = [i for i in range(self.N_system_qubits, self.N_ancilla_qubits + self.N_system_qubits)]

        qubits_to_measure = (qubits[q_No] for q_No in list(qubit_system_list) + qubit_ancilla_list)
        if qubit_system_list != []:
            yield cirq.measure(*qubits_to_measure)
        else:
            return None

    def _circuit_diagram_info_(self, args):
        string_list = []
        for _ in range(self.N_system_qubits):
            string_list.append(' Measuring system qubits')

        for _ in range(self.N_ancilla_qubits):
            string_list.append(' Measuring ancilla qubits')

        return string_list

    def num_qubits(self):
        return self.N_ancilla_qubits + self.N_system_qubits


from quchem.Qcircuit.Circuit_functions_to_create_arb_state import prepare_arb_state_IBM_to_cirq, prepare_arb_state_cirq_matrix_gate
def Get_LCU_G_circuit(method, state_vector, start_qubit_ind, check_G_circuit=True, allowed_qiskit_gates=['id', 'rz', 'ry', 'rx', 'cx' ,'s', 'h', 'y','z', 'x'], qiskit_opt_level=0):
    """
    class to generate G circuit of linear combination of unitaries technique

    Args:
        method (str): String of method used to build G circuit
                      Choices are: * IBM : uses IBM code to generate required circuit
                                   * matrix: builds perfect gate out of matrix
                                   * cirq_disentangle

        state_vector (list): list of amplitudes
        start_qubit_ind (int): start qubit index for circuit
        check_G_circuit (bool): Check whether circuit creates correct state
        allowed_qiskit_gates (list): list of strings of allowed IBM gates

    Returns
        A cirq circuit

    ** Example **

    import cirq
    import numpy as np

    state_vector = [0.5 , 0.5, 0.5, 0.5]
    start_qubit_ind = 2


    method = 'IBM'
    IBM_circuit = Get_LCU_G_circuit(method, state_vector, start_qubit_ind)
    print(IBM_circuit)


            2: ───Ry(0.5π)───X───Ry(0)───X───
                             │           │
            3: ───Ry(0.5π)───@───────────@───


    method = 'matrix'
    matrix_circuit = Get_LCU_G_circuit(method, state_vector, start_qubit_ind)
    print(matrix_circuit)

                  ┌                                           ┐
                  │ 0.5  +0.j -0.289+0.j -0.408+0.j -0.707+0.j│
            2: ───│ 0.5  +0.j  0.866+0.j  0.   +0.j  0.   +0.j│───
                  │ 0.5  +0.j -0.289+0.j  0.816+0.j  0.   +0.j│
                  │ 0.5  +0.j -0.289+0.j -0.408+0.j  0.707+0.j│
                  └                                           ┘
                  │
            3: ───#2──────────────────────────────────────────────

    method = 'cirq_disentangle'
    cirq_method_circuit = Get_LCU_G_circuit(method, state_vector, start_qubit_ind)
    print(cirq_method_circuit)


            2: ───Ry(0.5π)───X───Ry(0)───X───
                             │           │
            3: ───Ry(0.5π)───@───────────@───

    """

    if method == 'IBM':
        G_circuit = prepare_arb_state_IBM_to_cirq(state_vector, opt_level=qiskit_opt_level,allowed_gates=allowed_qiskit_gates, start_qubit_ind=start_qubit_ind)
    elif method == 'matrix':
        G_circuit = prepare_arb_state_cirq_matrix_gate(state_vector, start_qubit_ind=start_qubit_ind)
    elif method == 'cirq_disentangle':
        G_circuit, Global_phase = intialization_circuit(state_vector, start_qubit_ind, check_circuit=False)
    else:
        raise ValueError(f'unknown method to build G: {method}')

    if check_G_circuit:
        circuit_final_state = G_circuit.final_state_vector(ignore_terminal_measurements=True)

        if method == 'cirq_disentangle':
            circuit_final_state=circuit_final_state*Global_phase

        if not np.allclose(circuit_final_state, state_vector):
            overlap = np.dot(circuit_final_state,state_vector)
            print(f'Overlap between state_vecotr and final_circuit_state: {overlap: 1.5f}')
            raise ValueError('G circuit not preparing correct state from |00...0> start state')

    if method == 'cirq_disentangle':
        return G_circuit, Global_phase
    else:
        return G_circuit

from quchem.Unitary_Partitioning.Unitary_partitioning_LCU_method import Get_R_op_list
from functools import reduce
def Build_GUG_LCU_circuit(anti_commuting_set,
                          N_index, 
                          N_system_qubits,
                          G_method,
                          check_G_circuit=True,
                          allowed_qiskit_gates=['id', 'rz', 'ry', 'rx', 'cx' ,'s', 'h', 'y','z', 'x'], 
                          qiskit_opt_level=0, 
                          check_GUG_circuit=True,
                          check_Rl_reduction_lin_alg=True):
    """
    class to generate G circuit of linear combination of unitaries technique

    Args:
        anti_commuting_set(list): list of anti-commuting Qubit operators
        N_index (int): index of term in anti_commuting_set to reduce too
        G_method (str): String of method used to build G circuit
                      Choices are: * IBM : uses IBM code to generate required circuit
                                   * matrix: builds perfect gate out of matrix
                                   * cirq_disentangle

        check_G_circuit (bool): Check whether circuit creates correct state
        allowed_qiskit_gates (list): list of strings of allowed IBM gates
        check_GUG_circuit (bool): check GUG circuit encoding of operator (uses POVM to enforce all zero ancilla measurement)
        check_Rl_reduction_lin_alg (bool): check that R Hl R_dag results in only Pn
    Returns
        A cirq circuit

    ** Example **

    import cirq
    import numpy as np

    state_vector = [0.5 , 0.5, 0.5, 0.5]
    start_qubit_ind = 2


    method = 'IBM'
    IBM_circuit = Get_LCU_G_circuit(method, state_vector, start_qubit_ind)
    print(IBM_circuit)


            2: ───Ry(0.5π)───X───Ry(0)───X───
                             │           │
            3: ───Ry(0.5π)───@───────────@───


    method = 'matrix'
    matrix_circuit = Get_LCU_G_circuit(method, state_vector, start_qubit_ind)
    print(matrix_circuit)

                  ┌                                           ┐
                  │ 0.5  +0.j -0.289+0.j -0.408+0.j -0.707+0.j│
            2: ───│ 0.5  +0.j  0.866+0.j  0.   +0.j  0.   +0.j│───
                  │ 0.5  +0.j -0.289+0.j  0.816+0.j  0.   +0.j│
                  │ 0.5  +0.j -0.289+0.j -0.408+0.j  0.707+0.j│
                  └                                           ┘
                  │
            3: ───#2──────────────────────────────────────────────

    method = 'cirq_disentangle'
    cirq_method_circuit = Get_LCU_G_circuit(method, state_vector, start_qubit_ind)
    print(cirq_method_circuit)


            2: ───Ry(0.5π)───X───Ry(0)───X───
                             │           │
            3: ───Ry(0.5π)───@───────────@───

    """


    # 1. Get Rl as linear comb unitaries
    R_linear_comb_UNCORRECTED_QubitOpList, Pn, gamma_l = Get_R_op_list(anti_commuting_set, N_index, N_system_qubits, check_reduction=check_Rl_reduction_lin_alg, atol=1e-8, rtol=1e-05)

    #2. Remove any phases
    R_linear_comb_phase_corrcted_QubitOpList, R_linear_comb_correction_values, ancilla_amplitudes, l1_norm = absorb_complex_phases(R_linear_comb_UNCORRECTED_QubitOpList)


    #3. fill any missing ancilla amps with ZERO amplitude
    N_ancilla = np.log2(len(ancilla_amplitudes))
    if np.ceil(N_ancilla) != np.floor(N_ancilla):
        N_ancilla = int(np.ceil(N_ancilla))
        full_ancilla = np.zeros(2**N_ancilla)
        full_ancilla[:len(ancilla_amplitudes)] = np.array(ancilla_amplitudes)
        ancilla_amplitudes= full_ancilla.tolist()
    N_ancilla = int(np.log2(len(ancilla_amplitudes)))
    

    #4. build GUG circuit
    if G_method == 'cirq_disentangle':
        G_circ, Global_phase = Get_LCU_G_circuit(G_method, ancilla_amplitudes, N_system_qubits, check_G_circuit=check_G_circuit, allowed_qiskit_gates=allowed_qiskit_gates, qiskit_opt_level=qiskit_opt_level)
    else:
        G_circ = Get_LCU_G_circuit(G_method, ancilla_amplitudes, N_system_qubits, check_G_circuit=check_G_circuit, allowed_qiskit_gates=allowed_qiskit_gates, qiskit_opt_level=qiskit_opt_level)
        Global_phase=1

    G_dagger_circ = cirq.inverse(G_circ)


    U_circ_obj = LCU_U_gate(N_ancilla, N_system_qubits, R_linear_comb_phase_corrcted_QubitOpList, R_linear_comb_correction_values, Pn)
    U_circ_circ = cirq.Circuit(
        cirq.decompose_once((U_circ_obj(*cirq.LineQubit.range(U_circ_obj.num_qubits())))))


    R_circ_circ = cirq.Circuit([
        *G_circ.all_operations(),
        *U_circ_circ.all_operations(),
        *G_dagger_circ.all_operations(),
    ])

    if check_GUG_circuit:

        ## linear algebra
        Rl_QubitOp = reduce(lambda Op1, Op2: Op1+Op2, R_linear_comb_UNCORRECTED_QubitOpList)
        Rl_mat = qubit_operator_sparse(Rl_QubitOp, n_qubits=N_system_qubits)
        I_sys = np.eye(2**N_system_qubits)

        # #### circuit linear algebra, with POVM onto ancilla zero state
        zero_qubit_state = np.array([[1],[0]])
        aniclla_0_state = reduce(np.kron, [zero_qubit_state for _ in range(N_ancilla)])
        
        ket = np.kron(I_sys, aniclla_0_state)
        bra = ket.conj().T

        G_U_Gdag_mat = R_circ_circ.unitary(qubits_that_should_be_present = cirq.LineQubit.range(0, N_ancilla+N_system_qubits)) # important to specify all qubits present (errors where Identity qubits ignored if not specified)
        G_U_Gdag_mat = Global_phase * G_U_Gdag_mat

        traced_R = bra @ G_U_Gdag_mat @ ket *l1_norm

        # #### alternate way to do POVM:
        # ancilla_0_state = reduce(np.kron, [np.array([[1],[0]]) for _ in range(N_ancilla)])
        # ancilla_0_projector = np.outer(ancilla_0_state, ancilla_0_state)
        # POVM_0_ancilla = np.kron(I_sys, ancilla_0_projector) # forces all zero measurement on ancilla!
        # G_U_Gdag_mat = R_circ_circ.unitary()
        # projected_GUG = POVM_0_ancilla.dot(G_U_Gdag_mat)

        # trace_GUG = projected_GUG.reshape([2 ** N_system_qubits, 2 ** N_ancilla,
        #                                                 2 ** N_system_qubits, 2 ** N_ancilla])
        # traced_R = np.einsum('jiki->jk', trace_GUG)
        # traced_R = traced_R*l1_norm

        if not np.allclose(Rl_mat.todense(), traced_R):
            raise ValueError('GUG encoding not giving Rl operation')

    return R_circ_circ, Pn, gamma_l, l1_norm, N_ancilla


def Full_LCU_Rl_Circuit(anti_commuting_set,
                          N_index, 
                          N_system_qubits,
                          G_method,
                          ansatz_circ,
                          check_G_circuit=True,
                          allowed_qiskit_gates=['id', 'rz', 'ry', 'rx', 'cx' ,'s', 'h', 'y','z', 'x'], 
                          qiskit_opt_level=0, 
                          check_GUG_circuit=True,
                          check_Rl_reduction_lin_alg=True):
    """
    class to generate full cirq circuit that prepares a quantum state (ansatz) performs R operation as LCU and finally
    measures both the system and ancilla qubit lines in the Z basis.

    Args:
        anti_commuting_set(list): list of anti-commuting Qubit operators
        N_index (int): index of term in anti_commuting_set to reduce too
        G_method (str): String of method used to build G circuit
                      Choices are: * IBM : uses IBM code to generate required circuit
                                   * matrix: builds perfect gate out of matrix
                                   * cirq_disentangle
        
        ansatz_circ (cirq.Circuit): ansatz cirq Circuit
        check_G_circuit (bool): Check whether circuit creates correct state
        allowed_qiskit_gates (list): list of strings of allowed IBM gates
        check_GUG_circuit (bool): check GUG circuit encoding of operator (uses POVM to enforce all zero ancilla measurement)
        check_Rl_reduction_lin_alg (bool): check that R Hl R_dag results in only Pn
    Returns
        A cirq circuit
    """


    # GUG circuit of Rl
    R_circ_circ, Pn, gamma_l, l1_norm, N_ancilla =Build_GUG_LCU_circuit(anti_commuting_set,
                                          N_index, 
                                          N_system_qubits,
                                          G_method,
                                          check_G_circuit=check_G_circuit,
                                          allowed_qiskit_gates=allowed_qiskit_gates, 
                                          qiskit_opt_level=qiskit_opt_level, 
                                          check_GUG_circuit=check_GUG_circuit,
                                          check_Rl_reduction_lin_alg=check_Rl_reduction_lin_alg)

    # Get circuit to measure Pn in Z basis
    change_to_Z_basis_obj = Change_PauliWord_measurement_to_Z_basis(Pn)
    change_to_Z_basis_circ = cirq.Circuit(
        cirq.decompose_once((change_to_Z_basis_obj(*cirq.LineQubit.range(change_to_Z_basis_obj.num_qubits())))))

    # measure system and ancilla registers
    measure_obj = Measure_system_and_ancilla(Pn, N_ancilla, N_system_qubits)
    measure_obj_circ = cirq.Circuit(
        cirq.decompose_once((measure_obj(*cirq.LineQubit.range(measure_obj.num_qubits())))))

    full_Q_circ = cirq.Circuit([
        *ansatz_circ.all_operations(),
        *R_circ_circ.all_operations(),
        *change_to_Z_basis_circ.all_operations(),
        *measure_obj_circ
    ])
    return full_Q_circ, Pn, gamma_l, l1_norm, N_ancilla

def Get_Histogram_key_ancilla_system(qubitOperator, N_system_qubits, N_ancilla_qubits):
    """
    Function to give histogram key string of BOTH system and ancilla qubit lines.

    Args:
        qubitOperator (QubitOperator): Qubit operator to measure
        N_system_qubits (int): Number of system qubits / number of spin orbitals
        N_ancilla_qubits (int): Number of ancilla qubits

    Returns
        histogram_string (str): string of histogram key

    example

    x = QubitOperator('X0 Z2 Y3', 0.25j)
    N_ancilla_qubits = 2
    N_system_qubits = 5

    output = '0,2,3,5,6'


    """
    qubit_No, PauliStr = zip(*list(*qubitOperator.terms.keys()))

    histogram_string = ','.join([str(i) for i in (qubit_No)] + [str(i) for i in range(N_system_qubits, N_system_qubits + N_ancilla_qubits)])
    return histogram_string

def simulate_probabilistic_Q_circuit(probability_of_success, Quantum_circuit, n_shots):

    simulator = cirq.Simulator()
    raw_result = simulator.run(Quantum_circuit, repetitions=n_shots*int(np.ceil(1/probability_of_success)))
    return raw_result

def Get_binary_dict_project(Quantum_circuit, qubitOperator, n_shots, N_system_qubits, ancilla_amplitudes, l1_norm):
    N_system_terms_measured = len(list(qubitOperator.terms.keys())[0])
    N_ancilla_qubits = int(np.ceil(np.log2(len(ancilla_amplitudes))))
    correct_ancilla_state = np.zeros([N_ancilla_qubits])

    P_success = (1 / l1_norm) ** 2

    n_success_shots = 0
    binary_results_dict = {}
    while n_success_shots != n_shots:
        hist_key = Get_Histogram_key_ancilla_system(qubitOperator, N_system_qubits, N_ancilla_qubits)
        raw_result = simulate_probabilistic_Q_circuit(P_success, Quantum_circuit, n_shots)

        M_results = raw_result.measurements[hist_key]
        for result in M_results:

            if np.array_equal(result[N_system_terms_measured::],
                              correct_ancilla_state):  # Checks if all zero ancilla measured!
                seperator = ''
                state_key_binary = seperator.join(
                    map(str, result[:N_system_terms_measured]))  # Gets rid of ancilla part!!!
                if state_key_binary not in binary_results_dict.keys():
                    binary_results_dict[state_key_binary] = 1
                else:
                    binary_results_dict[state_key_binary] += 1
                n_success_shots += 1

            if n_success_shots == n_shots:
                break
    return binary_results_dict


#### linear algebra cirq circuit experiments
class LCU_VQE_Experiment_UP_circuit_lin_alg():

    def __init__(self, anti_commuting_sets, ansatz_circuit, N_system_qubits, G_method, N_indices_dict=None, check_G_circuit=True,
                  allowed_qiskit_gates=['id', 'rz', 'ry', 'rx', 'cx' ,'s', 'h', 'y','z', 'x'], 
                  qiskit_opt_level=0, 
                  check_GUG_circuit=True,
                  check_Rl_reduction_lin_alg=True):

        self.anti_commuting_sets = anti_commuting_sets
        self.ansatz_circuit = ansatz_circuit
        self.N_system_qubits = N_system_qubits
        self.ansatz_vector = ansatz_circuit.final_state_vector(ignore_terminal_measurements=True)
        self.G_method = G_method

        self.N_indices_dict = N_indices_dict
        self.check_G_circuit = check_G_circuit
        self.allowed_qiskit_gates=allowed_qiskit_gates
        self.qiskit_opt_level=0
        self.check_GUG_circuit=check_GUG_circuit
        self.check_Rl_reduction_lin_alg=check_Rl_reduction_lin_alg

        self.qubit_zero_state = np.array([[1],[0]])
        self.I_system_operation = np.eye(2**N_system_qubits)

    def Calc_Energy(self):

        E_list = []
        for set_key in self.anti_commuting_sets:

            anti_commuting_set = self.anti_commuting_sets[set_key]

            if len(anti_commuting_set) > 1:

                if self.N_indices_dict is None:
                    Q_circuit, Pn, gamma_l, l1_norm, N_ancilla = Full_LCU_Rl_Circuit(anti_commuting_set,
                                                                                          0,  # <- N_index set to 0 , 
                                                                                          self.N_system_qubits,
                                                                                          self.G_method,
                                                                                          self.ansatz_circuit,
                                                                                          check_G_circuit=self.check_G_circuit,
                                                                                          allowed_qiskit_gates=self.allowed_qiskit_gates, 
                                                                                          qiskit_opt_level=self.qiskit_opt_level, 
                                                                                          check_GUG_circuit=self.check_GUG_circuit,
                                                                                          check_Rl_reduction_lin_alg=self.check_Rl_reduction_lin_alg)
                else:
                    Q_circuit, Pn, gamma_l, l1_norm, N_ancilla = Full_LCU_Rl_Circuit(anti_commuting_set,
                                                                                          self.N_indices_dict[set_key],  # <- N_index set by Dict 
                                                                                          self.N_system_qubits,
                                                                                          self.G_method,
                                                                                          self.ansatz_circuit,
                                                                                          check_G_circuit=self.check_G_circuit,
                                                                                          allowed_qiskit_gates=self.allowed_qiskit_gates, 
                                                                                          qiskit_opt_level=self.qiskit_opt_level, 
                                                                                          check_GUG_circuit=self.check_GUG_circuit,
                                                                                          check_Rl_reduction_lin_alg=self.check_Rl_reduction_lin_alg)

                # use POVM onto all zero ancilla state
                circuit_matrix = Q_circuit.unitary(ignore_terminal_measurements=True)
                ancilla_0_state = reduce(np.kron, [self.qubit_zero_state for _ in range(N_ancilla)])


                #################### METHOD 1
                # ancilla_0_projector = np.outer(ancilla_0_state, ancilla_0_state)
                # POVM_0_ancilla = np.kron(self.I_system_operation, ancilla_0_projector) # forces all zero measurement on ancilla!

                ### project ancilla onto all zero state and leave system untouched
                # projected_circuit = POVM_0_ancilla.dot(circuit_matrix)

                # partial_trace_mat = projected_circuit.reshape([2 ** self.N_system_qubits, 2 ** N_ancilla,
                #                                                 2 ** self.N_system_qubits, 2 ** N_ancilla])
                
                # # trace out ancilla register
                # partial_trace = np.einsum('jiki->jk', partial_trace_mat) * l1_norm

                
                #################### METHOD 2

                ### project ancilla onto all zero state and leave system untouched
                ket_Isys_POVM_ancilla = np.kron(self.I_system_operation, ancilla_0_state)
                bra_Isys_POVM_ancilla = ket_Isys_POVM_ancilla.conj().T
                # POVM on ancilla then trace out ancilla register
                partial_trace = bra_Isys_POVM_ancilla @ circuit_matrix @ ket_Isys_POVM_ancilla *l1_norm
                #################### (EITHER USE METHOD 1 OR 2)

                ### USE partial_trace where ancilla all zero measurement forced
                final_state_ket = partial_trace[:,0]


                # note Q_circuit HAS change of basis for Pn! hence measure Z op version now
                PauliStr_Pn, beta_N= tuple(*Pn.terms.items())
                PauliStr_Pn_Z = [(qNo, 'Z')for qNo, Pstr in PauliStr_Pn]
                Pn = QubitOperator(PauliStr_Pn_Z, beta_N)
                Pn_matrix = qubit_operator_sparse(Pn, n_qubits=self.N_system_qubits)


                exp_result = np.trace(np.outer(final_state_ket, final_state_ket)@Pn_matrix)
                E_list.append(exp_result * gamma_l)
                # final_state_bra = final_state_ket.transpose().conj()
                # exp_result = final_state_bra.dot(Pn_matrix.todense().dot(final_state_ket))
                # E_list.append(exp_result.item(0) * gamma_l)


            else:
                qubitOp = anti_commuting_set[0]

                for PauliWord, const in qubitOp.terms.items():
                    if PauliWord != ():

                        final_state_ket = circuit_matrix.dot(self.ansatz_vector)
                        final_state_bra = final_state_ket.transpose().conj()

                        P_matrix = qubit_operator_sparse(qubitOp, n_qubits=self.n_qubits)

                        exp_result = final_state_bra.dot(P_matrix.todense().dot(final_state_ket))
                        E_list.append(exp_result.item(0) * const)

                    else:
                        E_list.append(const)

        return sum(E_list).real





###### TODO: functions not working below
class LCU_VQE_Experiment_UP_circuit_sampling():
    """
    TODO doc_string
    """
    def __init__(self,
                 anti_commuting_sets,
                 ansatz_circuit,
                 n_shots,
                 N_system_qubits,
                 N_indices_dict=None):

        self.anti_commuting_sets = anti_commuting_sets
        self.ansatz_circuit = ansatz_circuit
        self.n_shots = n_shots

        self.N_system_qubits = N_system_qubits
        self.N_indices_dict = N_indices_dict

    def Get_Histogram_key_ancilla_system(self, qubitOperator, N_ancilla_qubits):

        qubit_No, PauliStr = zip(*list(*qubitOperator.terms.keys()))
        histogram_string = ','.join([str(i) for i in (qubit_No)] + [str(i) for i in range(self.N_system_qubits,
                                                                                          self.N_system_qubits + N_ancilla_qubits)])
        return histogram_string

    def simulate_probabilistic_Q_circuit(self, probability_of_success, Quantum_circuit):
        simulator = cirq.Simulator()
        raw_result = simulator.run(Quantum_circuit, repetitions=10*self.n_shots * int(np.ceil(1 / probability_of_success)))
        # TODO note extra 1000 here in no.  of shots (due to only certain exp results taken when projected)
        # TODO could make an optional parameter
        return raw_result

    def Get_binary_dict_project(self, Quantum_circuit, qubitOperator, ancilla_amplitudes, l1_norm):
        N_system_terms_measured = len(list(qubitOperator.terms.keys())[0])
        N_ancilla_qubits = int(np.ceil(np.log2(len(ancilla_amplitudes))))
        correct_ancilla_state = np.zeros([N_ancilla_qubits])

        P_success = (1 / l1_norm) ** 2

        total_number_repeats = 0
        n_success_shots = 0
        binary_results_dict = {}
        while n_success_shots != self.n_shots:
            hist_key = Get_Histogram_key_ancilla_system(qubitOperator, self.N_system_qubits, N_ancilla_qubits)
            raw_result = simulate_probabilistic_Q_circuit(P_success, Quantum_circuit, self.n_shots)

            M_results = raw_result.measurements[hist_key]
            for result in M_results:
                total_number_repeats += 1
                if np.array_equal(result[N_system_terms_measured::],
                                  correct_ancilla_state):  # Checks if all zero ancilla measured!
                    seperator = ''
                    state_key_binary = seperator.join(
                        map(str, result[:N_system_terms_measured]))  # Gets rid of ancilla part!!!
                    if state_key_binary not in binary_results_dict.keys():
                        binary_results_dict[state_key_binary] = 1
                    else:
                        binary_results_dict[state_key_binary] += 1
                    n_success_shots += 1

                if n_success_shots == self.n_shots:
                    break
        return binary_results_dict, total_number_repeats

    def Calc_Energy(self, check_LCU_reduction=False):

        E_list = []
        number_of_circuit_evals = {}
        for set_key in self.anti_commuting_sets:
            if len(self.anti_commuting_sets[set_key]) > 1:

                if self.N_indices_dict is None:
                    R_uncorrected, Pn, gamma_l = Get_R_op_list(self.anti_commuting_sets[set_key], 0,
                                                               check_operator=check_LCU_reduction)
                    R_corrected_Op_list, R_corr_list, ancilla_amplitudes, l1_norm = absorb_complex_phases(R_uncorrected)
                else:
                    R_uncorrected, Pn, gamma_l = Get_R_op_list(self.anti_commuting_sets[set_key],
                                                                          self.N_indices_dict[set_key],
                                                               check_operator=check_LCU_reduction)
                    R_corrected_Op_list, R_corr_list, ancilla_amplitudes, l1_norm = absorb_complex_phases(R_uncorrected)

                Q_circuit = Full_Q_Circuit(Pn, R_corrected_Op_list, R_corr_list, ancilla_amplitudes,
                                           self.N_system_qubits, self.ansatz_circuit)

                binary_state_counter, No_circuit_M = self.Get_binary_dict_project(Q_circuit, Pn, ancilla_amplitudes, l1_norm)
                number_of_circuit_evals[set_key] = No_circuit_M
                exp_result = expectation_value_by_parity(binary_state_counter)

                E_list.append(exp_result * gamma_l)
                #print(Pn, gamma_l, exp_result, l1_norm)

            else:
                single_PauliOp = self.anti_commuting_sets[set_key][0]
                if list(single_PauliOp.terms.keys())[0] == ():
                    E_list.append(list(single_PauliOp.terms.values())[0])
                else:
                    Q_circuit = Generate_Full_Q_Circuit(self.ansatz_circuit, single_PauliOp)
                    hist_key_str = Get_Histogram_key(single_PauliOp)
                    int_state_counter = Simulate_Quantum_Circuit(Q_circuit, self.n_shots, hist_key_str)
                    number_of_circuit_evals[set_key] = self.n_shots
                    binary_state_counter = Return_as_binary(int_state_counter, hist_key_str)
                    exp_result = expectation_value_by_parity(binary_state_counter)
                    E_list.append(exp_result * list(single_PauliOp.terms.values())[0])
                    #print(single_PauliOp, exp_result * list(single_PauliOp.terms.values())[0])

        #         print(Q_circuit.to_text_diagram(transpose=True))
        return sum(E_list), number_of_circuit_evals

    def Get_wavefunction_of_state(self, sig_figs=3):
        return Get_wavefunction(self.ansatz_circuit, sig_figs=sig_figs)
