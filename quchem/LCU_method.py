from openfermion.ops import QubitOperator
import numpy as np

from quchem.Unitary_partitioning import *



def Get_R_op_list(anti_commuting_set, N_index, check_operator=False):
    """

    Function gets the R operator as a linear combination of unitary operators.

    First the X operator is found:
    X = i ∑_{k=0} 𝛿_{k} P_{kn}

    R has the definition:
    𝑅=exp(−𝑖𝛼X/2)=cos(𝛼/2)𝟙−𝑖sin(𝛼/2)X
    this is used to build R

    ###
     anti_set = ∑_{i=0} 𝛼_{i} P_{i}.
     normalised = 𝛾_{𝑙} ∑_{i=0} 𝛽_{i} P_{i}... where ∑_{i=0} 𝛽_{i}^{2} =1

     the 𝛽n Pn is then removed and set normalised again:
     H_{n_1} =  Ω_{𝑙} ∑_{k=0} 𝛿_{k} P_{k} ... where k!=n

    then:
    X = i ∑_{k=0} 𝛿_{k} P_{k} P_{n} = i ∑_{k=0} 𝛿_{k} P_{kn}
    ####
    Paper also defines
    H_n = cos(𝜙_{n-1}) Pn + sin(𝜙_{n-1}) H_{n_1 }

    currently have:
    H_{n}/𝛾_{𝑙} = 𝛽n Pn +  Ω_{𝑙} H_{n_1}

    therefore:
    𝜙_{n-1} = arccos(𝛽n)
    as Ω_{𝑙} is always positive, so if 𝜙_{n-1} > 𝜋 ....THEN.... 𝜙_{n-1} = 2𝜋 - arccos(𝛽n)


    Args:
        anti_commuting_set (list): list of anti-commuting qubit operators
        N_index (int): index of term to reduce too
    Returns:
        R_linear_comb_list (list): linear combination of R operators that makes up R operator
        P_n: (QubitOperator): qubit operator to be reduced too (Pn)
        gamma_l (float): normalisation term (𝛾_{𝑙])
     """

    # 𝛾_𝑙 ∑ 𝛽_𝑗 𝑃_𝑗
    normalised_FULL_set = Get_beta_j_cofactors(anti_commuting_set)
    gamma_l = normalised_FULL_set['gamma_l']

    norm_FULL_set = normalised_FULL_set['PauliWords'].copy()

    # 𝛽_n 𝑃_n
    qubitOp_Pn_beta_n = norm_FULL_set.pop(N_index)

    # Ω_𝑙 ∑ 𝛿_k 𝑃_k  ... note this doesn't contain 𝛽_n 𝑃_n
    H_n_1 = Get_beta_j_cofactors(norm_FULL_set)
    Omega_l = H_n_1['gamma_l']

    ##

    # cos(𝜙_{𝑛−1}) =𝛽_𝑛
    phi_n_1 = np.arccos(list(qubitOp_Pn_beta_n.terms.values())[0])

    # require sin(𝜙_{𝑛−1}) to be positive...
    # this uses CAST diagram to ensure the sign term is positive and cos term has correct sign (can be negative)
    if (phi_n_1 > np.pi):
        # ^ as sin phi_n_1 must be positive phi_n_1 CANNOT be larger than 180 degrees!
        phi_n_1 = 2 * np.pi - phi_n_1
        print('correct quadrant found!!!')

    #     𝑅=exp(−𝑖𝛼 X/2)=cos(𝛼/2)𝟙 − 𝑖sin(𝛼/2)X = cos(𝛼/2)𝟙 − 𝑖sin(𝛼/2)(i∑𝛿𝑘 𝑃𝑘𝑃𝑛)
    #     𝑅=exp(−𝑖𝛼 X/2)=cos(𝛼/2)𝟙 − 𝑖sin(𝛼/2)X = cos(𝛼/2)𝟙 + sin(𝛼/2)(∑𝛿𝑘 𝑃𝑘𝑃𝑛) #<--- note sign here!
    Pn = QubitOperator(list(qubitOp_Pn_beta_n.terms.keys())[0],
                       1)  # np.sign(list(qubitOp_Pn_beta_n.terms.values())[0]))

    alpha = phi_n_1.copy()
    #     print('alpha/2 =', (alpha/(2*np.pi))*360/2)

    I_term = QubitOperator('', np.cos(alpha / 2))
    R_linear_comb_list = [I_term]

    sin_term = -np.sin(alpha / 2)

    for qubitOp_Pk in H_n_1['PauliWords']:
        PkPn = qubitOp_Pk * Pn
        R_linear_comb_list.append(sin_term * PkPn)

    if not np.isclose(sum(np.absolute(list(qubitOp.terms.values())[0]) ** 2 for qubitOp in R_linear_comb_list), 1):
        raise ValueError(
            'normalisation of X operator incorrect: {}'.format(sum(np.absolute(list(qubitOp.terms.values())[0]) ** 2
                                                                   for qubitOp in R_linear_comb_list)))
    if check_operator:
    # #     # 𝐻𝑛= B𝑛𝑃𝑛+ Ω 𝑙∑𝛿𝑃𝑗
    # #     print('Hn =',qubitOp_Pn_beta_n, '+', Omega_l,' * ', H_n_1['PauliWords'])
    # #     #𝐻𝑛= cos(𝜙_{n-1}) Pn + sin(𝜙_{n-1}) H_{n_1 }
    # #     print('Hn =',np.cos(phi_n_1),Pn, '+', np.sin(phi_n_1),' * ', H_n_1['PauliWords'])
        Hn_list = [qubitOp_Pn_beta_n] + [Omega_l* op for op in  H_n_1['PauliWords']]
    #
    # #     print('')
    # #     print('R = ', R_linear_comb_list)
    # #     #R= cos(𝛼/2)𝟙-sin(𝛼/2)(∑𝛿_{𝑘}𝑃_{𝑘𝑛})
    # #     print('R = ', np.cos(alpha/2), 'I', '+',np.sin(alpha/2), [dkPk*Pn for dkPk in H_n_1['PauliWords']])

        ### CHECKING need to comment out as expensive!
        R = QubitOperator()
        for op in R_linear_comb_list:
            R += op

        R_dag = QubitOperator()
        for op in R:
            if list(op.terms.keys())[0]==():
                R_dag+= QubitOperator('', list(op.terms.values())[0])
            else:
                R_dag+=op*-1   #  note sign!!!

        H_n = QubitOperator()
        for op in Hn_list:
            H_n += op

        print('Pn= R*H_n*R_dag ')
        print('Pn=', Pn)
        print('R*H_n*R_dag = ', R * H_n * R_dag)
            # print('Pn= R*H_n*R_dag ', Pn, ' = ', R*H_n*R_dag)
        #     print('H_n= R_dag*Pn*R ', H_n, ' = ', R_dag*Pn*R)

    return R_linear_comb_list, Pn, gamma_l  # , H_n_1['PauliWords'], phi_n_1, Hn_list


def absorb_complex_phases(R_linear_comb_list):
    """
    Function absorbs phase of list of linear combination of unitary operators.

    ##
    Op = ∑_{i=0} 𝛼_{j} U_{j}

    used to find:
    l1_norm = ∑_{i=0} |𝛼_{𝑗}|

    all 𝛼_{j} made positive and phase correction held in R_linear_comb_correction_values list!
    ancilla_amplitudes = ∑_{i=0} (𝛼_{𝑗} / l1_norm)^{0.5} |j>

    Args:
        R_linear_comb_list (list): list of qubit operators

    Returns:
        R_linear_comb_correction_values (list): list of corrections to ensure all terms LCU sum are positive.
        R_linear_comb_corrected_phase (QubitOperator): list of qubit operators for LCU method
        ancilla_amplitudes: ancillar amplitudes for control U (LCU method)
        l1_norm (float): l1 norm

         """
    R_linear_comb_corrected_phase = []
    R_linear_comb_correction_values = []
    ancilla_amplitudes = []

    l1_norm = sum([np.absolute(const) for qubitOp in R_linear_comb_list for PauliWord, const in qubitOp.terms.items()])

    for qubitOp in R_linear_comb_list:
        for pauliword, const in qubitOp.terms.items():
            if (isinstance(const, complex)) and (const.imag < 0):
                R_linear_comb_corrected_phase.append(QubitOperator(pauliword, np.sqrt(np.absolute(const) / l1_norm)))
                R_linear_comb_correction_values.append(-1j)
                ancilla_amplitudes.append(np.sqrt(np.absolute(const) / l1_norm))  # .append(np.sqrt(const.imag**2))
            elif (isinstance(const, complex)) and (const.imag != 0):
                R_linear_comb_corrected_phase.append(QubitOperator(pauliword, np.sqrt(np.absolute(const) / l1_norm)))
                R_linear_comb_correction_values.append(1j)
                ancilla_amplitudes.append(np.sqrt(np.absolute(const) / l1_norm))  # .append(np.sqrt(const.imag**2))
            elif const < 0:
                R_linear_comb_corrected_phase.append(QubitOperator(pauliword, np.sqrt(np.absolute(const) / l1_norm)))
                R_linear_comb_correction_values.append(-1)
                ancilla_amplitudes.append(np.sqrt(np.absolute(const) / l1_norm))  # .append(np.sqrt(const**2))
            else:
                R_linear_comb_corrected_phase.append(QubitOperator(pauliword, np.sqrt(np.absolute(const) / l1_norm)))
                R_linear_comb_correction_values.append(1)
                ancilla_amplitudes.append(np.sqrt(np.absolute(const) / l1_norm))  # .append(np.sqrt(const**2))

    if not np.isclose(sum(np.absolute(amp) ** 2 for amp in ancilla_amplitudes), 1):
        raise ValueError('ancilla amplitudes NOT normalised properly')

    return R_linear_comb_corrected_phase, R_linear_comb_correction_values, ancilla_amplitudes, l1_norm

class Perform_modified_Pauligate(cirq.SingleQubitGate):
    """

    Function performs a pauligate multiplied by a constant.

    Args:
        PauliStr (str): string of Pauli operator to be performed
        correction_value (complex): constant to multiply gate by

    Returns
        A cirq circuit object to be used by cirq.Circuit

    example:
    1: ───(-0-1j)*X───

    matrix operation = array([  [0.-0.j, 0.-1.j],
                                [0.-1.j, 0.-0.j]])

    """

    def __init__(self, PauliStr, correction_value):

        self.PauliStr = PauliStr
        self.correction_value = correction_value

    def _unitary_(self):

        if self.PauliStr == 'Z':
            return cirq.Z._unitary_() * self.correction_value

        elif self.PauliStr == 'Y':
            return cirq.Y._unitary_() * self.correction_value

        elif self.PauliStr == 'X':
            return cirq.X._unitary_() * self.correction_value

        elif self.PauliStr == 'I':
            return cirq.I._unitary_() * self.correction_value

        else:
            raise TypeError('not a Pauli operation')

    def num_qubits(self):
        return 1

    def _circuit_diagram_info_(self, args):
        return '{}*{}'.format(self.correction_value, self.PauliStr)

class Perform_Modified_PauliWord(cirq.Gate):
    """
    Class to generate cirq circuit as a gate that performs a modified PauliWord

    Args:
        PauliQubitOp (QubitOperator): Qubit operator to perform as a circuit
        correction_val (complex): constant to multiply qubit operator by (intended to be phase factor for LCU)

    Returns
        A cirq circuit object to be used by cirq.Circuit

    Example:

    PauliQubitOp = QubitOperator('Y0 X1 X2 X3',1)
    correction_val = -1j

    output

    0: ───(-0-1j)*Y───
    1: ───X───────────
    2: ───X───────────
    3: ───X───────────

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
            yield Perform_modified_Pauligate('I', self.correction_val).on(qubits[self.sign_index])
        else:
            qubitNos_list, P_strs_list = zip(*list(self.PauliQubitOp.terms.keys())[0])

            for index, qNo in enumerate(qubitNos_list):
                P_str = P_strs_list[index]
                if qNo == self.sign_index:
                    yield Perform_modified_Pauligate(P_str, self.correction_val).on(qubits[qNo])
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

class LCU_R_gate(cirq.Gate):
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

        for control_state_index, R_qubitOp_corrected in enumerate(self.R_corrected_Op_list):

            if list(R_qubitOp_corrected.terms.keys())[0] != ():

                control_str = Get_state_as_str(self.No_control_qubits, control_state_index)
                control_values = [int(bit) for bit in control_str]

                qubitNos_list, _ = zip(*list(R_qubitOp_corrected.terms.keys())[0])

                qubit_list = cirq.LineQubit.range(self.No_system_qubits, self.No_system_qubits + self.No_control_qubits) \
                             + cirq.LineQubit.range(qubitNos_list[-1] + 1)  # note control qubits first!

                mod_p_word_gate = Perform_Modified_PauliWord(R_qubitOp_corrected,
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

                    mod_p_word_gate = Perform_Modified_PauliWord(R_qubitOp_corrected,
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

from quchem.quantum_circuit_functions import *

def Full_Q_Circuit(Pn, R_corrected_Op_list, R_correction_list, ancilla_amplitudes, N_system_qubits, ansatz_circ):

    """
    class to generate full cirq circuit that prepares a quantum state (ansatz) performs R operation as LCU and finally
    measures both the system and ancilla qubit lines in the Z basis.

    Args:
        Pn (QubitOperator): Qubit operator to measure
        R_corrected_Op_list (list): list of LCU qubit Operators
        R_corr_list (list): list of correction phases for LCU operators
        ancilla_amplitudes (list): list of ancilla amplitudes
        N_system_qubits (int): Number of system qubits / number of spin orbitals
        ansatz_circ (cirq.Circuit): cirq ansatz quantum circuit

    Returns
        A cirq circuit object to be used by cirq.Circuit

    example

    Pn= QubitOperator('Z3', 1)
    R_corrected_Op_list = [0.30200159443367586 [], 0.9533074199645766 [Y0 X1 X2 X3]]
    R_corr_list = [1, (-0-1j)]
    ancilla_amplitudes = [0.30200159443367586, 0.9533074199645766]
    N_system_qubits = 4
    ansatz_circ= Q_circuit

    output:

0: ───X─────────────────Rx(0.5π)───@─────────────────────────────────@───Rx(-0.5π)───I──────Y0──────────────────────────
                                   │                                 │               │      │
1: ───X─────────────────H──────────X───@─────────────────────────@───X───H───────────I──────X1──────────────────────────
                                       │                         │                   │      │
2: ───H────────────────────────────────X───@─────────────────@───X───H───────────────I──────X2──────────────────────────
                                           │                 │                       │      │
3: ───H────────────────────────────────────X───Rz(-1.947π)───X───H───────────────────1*I3───(-0-1j)*X3────────────────M─
                                                                                     │      │                         │
4: ─── U = 1.264 rad ────────────────────────────────────────────────────────────────(0)────@─── ─── U = 1.264 rad ───M─

    """

    ancilla_obj = prepare_arb_state(ancilla_amplitudes, N_system_qubits)
    ancilla_circ = ancilla_obj.Get_state_prep_Circuit()
    N_ancilla_qubits = int(np.ceil(np.log2(len(ancilla_amplitudes))))

    R_circ_obj = LCU_R_gate(N_ancilla_qubits, N_system_qubits, R_corrected_Op_list, R_correction_list, Pn)
    R_circ_circ = cirq.Circuit(
        cirq.decompose_once((R_circ_obj(*cirq.LineQubit.range(R_circ_obj.num_qubits())))))

    change_to_Z_basis_obj = Change_PauliWord_measurement_to_Z_basis(Pn)
    change_to_Z_basis_circ = cirq.Circuit(
        cirq.decompose_once((change_to_Z_basis_obj(*cirq.LineQubit.range(change_to_Z_basis_obj.num_qubits())))))

    measure_obj = Measure_system_and_ancilla(Pn, N_ancilla_qubits, N_system_qubits)

    measure_obj_circ = cirq.Circuit(
        cirq.decompose_once((measure_obj(*cirq.LineQubit.range(measure_obj.num_qubits())))))

    full_Q_circ = cirq.Circuit([
        *ansatz_circ.all_operations(),
        *ancilla_circ.all_operations(),
        *R_circ_circ.all_operations(),
        *list(ancilla_circ.all_operations())[::-1],
        *change_to_Z_basis_circ.all_operations(),
        *measure_obj_circ
    ])
    return full_Q_circ

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

from quchem.Simulating_Quantum_Circuit import *
class VQE_Experiment_LCU_UP():
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

        n_success_shots = 0
        binary_results_dict = {}
        while n_success_shots != self.n_shots:
            hist_key = Get_Histogram_key_ancilla_system(qubitOperator, self.N_system_qubits, N_ancilla_qubits)
            raw_result = simulate_probabilistic_Q_circuit(P_success, Quantum_circuit, self.n_shots)

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

                if n_success_shots == self.n_shots:
                    break
        return binary_results_dict

    def Calc_Energy(self, check_LCU_reduction=False):

        E_list = []
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

                binary_state_counter = self.Get_binary_dict_project(Q_circuit, Pn, ancilla_amplitudes, l1_norm)
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
                    binary_state_counter = Return_as_binary(int_state_counter, hist_key_str)
                    exp_result = expectation_value_by_parity(binary_state_counter)
                    E_list.append(exp_result * list(single_PauliOp.terms.values())[0])
                    #print(single_PauliOp, exp_result * list(single_PauliOp.terms.values())[0])

        #         print(Q_circuit.to_text_diagram(transpose=True))
        return sum(E_list)

    def Get_wavefunction_of_state(self, sig_figs=3):
        return Get_wavefunction(self.ansatz_circuit, sig_figs=sig_figs)


####
def Full_Ansatz_and_Quantum_R_circuit(Pn, R_corrected_Op_list, R_correction_list, ancilla_amplitudes, N_system_qubits,
                                      ansatz_circ, decompose_into_two_qubit_gates=False):

    """
    Generate cirq circuit doing the following:

     R U_{CC} U_{HF} |0〉_system x U_{LCU}|0〉_ancilla

    overall HF state, UCC ansatz and then R circuit with ancilla control performed.
    Args:
        Pn (QubitOperator): Qubit operator to measure
        R_corrected_Op_list (list): list of LCU qubit Operators
        R_corr_list (list): list of correction phases for LCU operators
        ancilla_amplitudes (list): list of ancilla amplitudes
        N_system_qubits (int): Number of system qubits / number of spin orbitals
        ansatz_circ (cirq.Circuit): cirq ansatz quantum circuit

    Returns
        A cirq circuit object to be used by cirq.Circuit

    example

    from openfermion.ops import QubitOperator
    Pn = QubitOperator('Z3', 1)
    R_corrected_Op_list = [ QubitOperator('',0.30200159443367586), QubitOperator('Y0 X1 X2 X3', 0.9533074199645766)]
    R_corr_list = [1, (-0 - 1j)]
    ancilla_amplitudes = [0.30200159443367586, 0.9533074199645766]
    N_system_qubits = 4
    ansatz_circ = Q_circuit #cirq.Circuit()

    Full_Ansatz_and_Quantum_R_circuit(Pn, R_corrected_Op_list, R_corr_list, ancilla_amplitudes, N_system_qubits,
                                      ansatz_circ, decompose_into_two_qubit_gates=True)

    output:

0: ───X─────────────────Rx(0.5π)───@─────────────────────────────────@───Rx(-0.5π)───I──────Y0──────────────────────
                                   │                                 │               │      │
1: ───X─────────────────H──────────X───@─────────────────────────@───X───H───────────I──────X1──────────────────────
                                       │                         │                   │      │
2: ───H────────────────────────────────X───@─────────────────@───X───H───────────────I──────X2──────────────────────
                                           │                 │                       │      │
3: ───H────────────────────────────────────X───Rz(-1.947π)───X───H───────────────────1*I3───(-0-1j)*X3──────────────
                                                                                     │      │
4: ─── U = 1.264 rad ────────────────────────────────────────────────────────────────(0)────@─────── U = 1.264 rad ─

    """



    if decompose_into_two_qubit_gates:
        ancilla_obj = prepare_arb_state_one_two_qubit_gates(ancilla_amplitudes, N_system_qubits) # <-- single and two qubit gate construction!
        ancilla_circ = ancilla_obj.Get_state_prep_Circuit()
        N_ancilla_qubits = int(np.ceil(np.log2(len(ancilla_amplitudes))))

        R_circ_obj = LCU_R_gate(N_ancilla_qubits, N_system_qubits, R_corrected_Op_list, R_correction_list, Pn)
        R_circ_circ = cirq.Circuit(
            cirq.decompose((R_circ_obj(*cirq.LineQubit.range(R_circ_obj.num_qubits())))))
    else:
        ancilla_obj = prepare_arb_state(ancilla_amplitudes, N_system_qubits)
        ancilla_circ = ancilla_obj.Get_state_prep_Circuit()
        N_ancilla_qubits = int(np.ceil(np.log2(len(ancilla_amplitudes))))

        R_circ_obj = LCU_R_gate(N_ancilla_qubits, N_system_qubits, R_corrected_Op_list, R_correction_list, Pn)
        R_circ_circ = cirq.Circuit(
            cirq.decompose_once((R_circ_obj(*cirq.LineQubit.range(R_circ_obj.num_qubits())))))


    full_Q_circ = cirq.Circuit([
        *ansatz_circ.all_operations(),
        *ancilla_circ.all_operations(),
        *R_circ_circ.all_operations(),
        *list(ancilla_circ.all_operations())[::-1],
#         *change_to_Z_basis_circ.all_operations(),
    ])
    return full_Q_circ

class VQE_Experiment_LCU_UP_lin_alg():
    """
    TODO doc_string
    """
    def __init__(self,
                 anti_commuting_sets,
                 ansatz_circuit,
                 N_system_qubits,
                 N_indices_dict=None):

        self.anti_commuting_sets = anti_commuting_sets
        self.ansatz_circuit = ansatz_circuit

        self.N_system_qubits = N_system_qubits
        self.N_indices_dict = N_indices_dict

        self.pauliDict=   {'X':np.array([[0,1],[1,0]]),
                          'Y':np.array([[0,-1j],[1j,0]]),
                          'Z':np.array([[1,0],[0,-1]]),
                          'I': np.eye(2)}
        # self.pauliDict=   {'X': cirq.H._unitary_(),
        #                   'Y': cirq.rx(np.pi / 2)._unitary_(),
        #                   'Z': np.eye(2),
        #                   'I': np.eye(2)}

        self.zero_state = np.array([[1], [0]])


    def Get_parital_system_density_matrix(self, Q_circuit_no_M_gates):

        input_state = [self.zero_state for _ in range(len(Q_circuit_no_M_gates.all_qubits()))]
        input_ket = reduce(kron, input_state)
        circuit_matrix = Q_circuit_no_M_gates.unitary()

        ansatz_state_ket = circuit_matrix.dot(input_ket.todense())

        full_density_matrix = np.outer(ansatz_state_ket, ansatz_state_ket)

        # simulator = cirq.Simulator()
        # output_ket = simulator.compute_amplitudes(Q_circuit_no_M_gates,
        #                                       bitstrings=[i for i in range(2 ** len(Q_circuit_no_M_gates.all_qubits()))])
        #
        # full_density_matrix = np.outer(output_ket, output_ket)

        ## First project state onto all zero ancilla state using POVM
        n_qubits = len(Q_circuit_no_M_gates.all_qubits())
        n_ancilla = n_qubits - self.N_system_qubits

        I_system_operator = np.eye((2**self.N_system_qubits))

        ancilla_0_state_list = [self.zero_state for _ in range(n_ancilla)]
        ancilla_0_state = reduce(np.kron, ancilla_0_state_list)
        ancilla_0_projector = np.outer(ancilla_0_state, ancilla_0_state)

        POVM_0_ancilla = np.kron(I_system_operator, ancilla_0_projector)
        Kraus_Op_0 = POVM_0_ancilla.copy()

        term = Kraus_Op_0.dot(full_density_matrix.dot(Kraus_Op_0.transpose().conj()))
        projected_density_matrix = term/np.trace(term) # projected into correct space using POVM ancilla measurement!

        ## Next get partial density matrix over system qubits # aka partial trace!
        # https://scicomp.stackexchange.com/questions/27496/calculating-partial-trace-of-array-in-numpy
        # reshape to do the partial trace easily using np.einsum
        reshaped_dm = projected_density_matrix.reshape([2 ** self.N_system_qubits, 2 ** n_ancilla,
                                                        2 ** self.N_system_qubits, 2 ** n_ancilla])
        reduced_dm = np.einsum('jiki->jk', reshaped_dm)

        # ### taking partial trace ### manual method!
        # # p_a = sum_{b} (I_{a}*<b|) p_{ab} (I_{a}*|b>)
        # basis_ancilla = np.eye((2 ** n_ancilla))
        # reduced_dm = np.zeros((2 ** self.N_system_qubits, 2 ** self.N_system_qubits), dtype=complex)
        # for b in range(basis_ancilla.shape[0]):
        #     b_ket = basis_ancilla[b, :].reshape([2 ** n_ancilla, 1])
        #     I_a_b_ket = np.kron(I_system_operator, b_ket)
        #     I_a_b_bra = I_a_b_ket.transpose().conj()
        #
        #     term = I_a_b_bra.dot(projected_density_matrix.dot(I_a_b_ket))
        #     reduced_dm += term

        if not np.isclose(np.trace(reduced_dm), 1):
            raise ValueError('partial density matrix is not normalised properly {}'.format(np.trace(reduced_dm)))

        return reduced_dm

    def Get_standard_ket(self):
        # simulator = cirq.Simulator()
        # output_ket = simulator.compute_amplitudes(self.ansatz_circuit,
        #                                       bitstrings=[i for i in range(2 ** len(self.ansatz_circuit.all_qubits()))])
        #
        input_state = [self.zero_state for _ in range(self.N_system_qubits)]
        input_ket = reduce(kron, input_state)
        circuit_matrix = self.ansatz_circuit.unitary()

        ansatz_state_ket = circuit_matrix.dot(input_ket.todense())

        if not np.isclose(sum([np.abs(i)**2 for i in ansatz_state_ket]), 1):
            raise ValueError('output ket is not normalised properly {}'.format(sum([np.abs(i)**2 for i in ansatz_state_ket])))

        return np.array(ansatz_state_ket) #.reshape([(2 ** len(self.ansatz_circuit.all_qubits())), 1])

    def Get_pauli_matrix(self, PauliOp):
        list_Q_nos, list_P_strs = list(zip(*[Paulistrs for Paulistrs, const in PauliOp.terms.items()][0]))

        list_of_ops = []
        # list_of_ops_print=[]
        for i in range(self.N_system_qubits):
            if i in list_Q_nos:
                index = list_Q_nos.index(i)
                list_of_ops.append(self.pauliDict[list_P_strs[index]])
                # list_of_ops_print.append('{}{}'.format(list_P_strs[index], i))
            else:
                list_of_ops.append(self.pauliDict['I'])
                # list_of_ops_print.append('I{}'.format(i))

        # print(list_of_ops_print, PauliOp)
        matrix = reduce(kron, list_of_ops)

        return matrix

    def Calc_Energy(self, check_LCU_reduction=False):
        # from openfermion.transforms import get_sparse_operator

        E_list = []
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

                ### checking ancilla line!
                N_ancilla_qubits = int(np.ceil(np.log2(len(ancilla_amplitudes))))
                ancilla_obj = prepare_arb_state(ancilla_amplitudes, self.N_system_qubits)  # perfect gate (but cannot decompose)
                # ancilla_obj = prepare_arb_state_one_two_qubit_gates(ancilla_amplitudes, self.N_system_qubits) #<-- single and two qubit gate construction!
                ancilla_circ = ancilla_obj.Get_state_prep_Circuit()

                simulator = cirq.Simulator()
                output_ket = simulator.compute_amplitudes(ancilla_circ,
                                                      bitstrings=[i for i in range(2 ** N_ancilla_qubits)])
                # print(ancilla_amplitudes)
                # print(output_ket[:len(ancilla_amplitudes)])
                # print(np.allclose(ancilla_amplitudes, output_ket[:len(ancilla_amplitudes)]))
                # print('###########')
                if not np.allclose(ancilla_amplitudes, output_ket[:len(ancilla_amplitudes)]):
                    print('MISTAKE ON ANCILLA LINE!!!!!!!!!!')



                # gives R|ψ〉
                Q_circuit = Full_Ansatz_and_Quantum_R_circuit(Pn, R_corrected_Op_list, R_corr_list, ancilla_amplitudes,
                                           self.N_system_qubits, self.ansatz_circuit)

                # print(Q_circuit.to_text_diagram(transpose=True))
                # print('')
                # print('##')

                partial_density_matrix = self.Get_parital_system_density_matrix(Q_circuit)

                H_sub_term_matrix = self.Get_pauli_matrix(Pn)
                # H_sub_term_matrix = get_sparse_operator(Pn, n_qubits=self.N_system_qubits)
                # # E=〈ψ | H | ψ〉= ∑_j  αj〈ψA | R† Pn R | ψA〉 #### where RQR = Pn

                # E= Tr(Pn rho)
                energy = np.trace(partial_density_matrix.dot(H_sub_term_matrix.todense()))
                # energy = np.trace(H_sub_term_matrix.dot(partial_density_matrix))
                E_list.append(energy * gamma_l)

            else:
                single_PauliOp = self.anti_commuting_sets[set_key][0]
                if list(single_PauliOp.terms.keys())[0] == ():
                    E_list.append(list(single_PauliOp.terms.values())[0])
                else:
                    ansatz_state_ket = self.Get_standard_ket()
                    ansatz_state_bra = ansatz_state_ket.transpose().conj()
                    # H_sub_term_matrix = get_sparse_operator(single_PauliOp, n_qubits=self.N_system_qubits)

                    # E=〈ψ | H | ψ〉= ∑_j  αj〈ψ | Pj | ψ〉
                    H_sub_term_matrix = self.Get_pauli_matrix(single_PauliOp)
                    # H_sub_term_matrix = get_sparse_operator(single_PauliOp, n_qubits=self.N_system_qubits)
                    energy = ansatz_state_bra.dot(H_sub_term_matrix.todense().dot(ansatz_state_ket))
                    E_list.append(energy.item(0) * list(single_PauliOp.terms.values())[0])

        return sum(E_list)

    def Get_wavefunction_of_ansatz_state(self, sig_figs=3):
        return Get_wavefunction(self.ansatz_circuit, sig_figs=sig_figs)


class VQE_Experiment_LCU_UP_lin_alg_single_two_qubit_gate():
    """
    TODO doc_string
    """
    def __init__(self,
                 anti_commuting_sets,
                 ansatz_circuit,
                 N_system_qubits,
                 N_indices_dict=None):

        self.anti_commuting_sets = anti_commuting_sets
        self.ansatz_circuit = ansatz_circuit

        self.N_system_qubits = N_system_qubits
        self.N_indices_dict = N_indices_dict

        self.pauliDict=   {'X':np.array([[0,1],[1,0]]),
                          'Y':np.array([[0,-1j],[1j,0]]),
                          'Z':np.array([[1,0],[0,-1]]),
                          'I': np.eye(2)}
        # self.pauliDict=   {'X': cirq.H._unitary_(),
        #                   'Y': cirq.rx(np.pi / 2)._unitary_(),
        #                   'Z': np.eye(2),
        #                   'I': np.eye(2)}

        self.zero_state = np.array([[1], [0]])


    def Get_parital_system_density_matrix(self, Q_circuit_no_M_gates):

        input_state = [self.zero_state for _ in range(len(Q_circuit_no_M_gates.all_qubits()))]
        input_ket = reduce(kron, input_state)
        circuit_matrix = Q_circuit_no_M_gates.unitary()

        ansatz_state_ket = circuit_matrix.dot(input_ket.todense())

        full_density_matrix = np.outer(ansatz_state_ket, ansatz_state_ket)

        # simulator = cirq.Simulator()
        # output_ket = simulator.compute_amplitudes(Q_circuit_no_M_gates,
        #                                       bitstrings=[i for i in range(2 ** len(Q_circuit_no_M_gates.all_qubits()))])
        #
        # full_density_matrix = np.outer(output_ket, output_ket)

        ## First project state onto all zero ancilla state using POVM
        n_qubits = len(Q_circuit_no_M_gates.all_qubits())
        n_ancilla = n_qubits - self.N_system_qubits

        I_system_operator = np.eye((2**self.N_system_qubits))

        ancilla_0_state_list = [self.zero_state for _ in range(n_ancilla)]
        ancilla_0_state = reduce(np.kron, ancilla_0_state_list)
        ancilla_0_projector = np.outer(ancilla_0_state, ancilla_0_state)

        POVM_0_ancilla = np.kron(I_system_operator, ancilla_0_projector)
        Kraus_Op_0 = POVM_0_ancilla.copy()

        term = Kraus_Op_0.dot(full_density_matrix.dot(Kraus_Op_0.transpose().conj()))
        projected_density_matrix = term/np.trace(term) # projected into correct space using POVM ancilla measurement!

        ## Next get partial density matrix over system qubits # aka partial trace!
        # https://scicomp.stackexchange.com/questions/27496/calculating-partial-trace-of-array-in-numpy
        # reshape to do the partial trace easily using np.einsum
        reshaped_dm = projected_density_matrix.reshape([2 ** self.N_system_qubits, 2 ** n_ancilla,
                                                        2 ** self.N_system_qubits, 2 ** n_ancilla])
        reduced_dm = np.einsum('jiki->jk', reshaped_dm)

        # ### taking partial trace ### manual method!
        # # p_a = sum_{b} (I_{a}*<b|) p_{ab} (I_{a}*|b>)
        # basis_ancilla = np.eye((2 ** n_ancilla))
        # reduced_dm = np.zeros((2 ** self.N_system_qubits, 2 ** self.N_system_qubits), dtype=complex)
        # for b in range(basis_ancilla.shape[0]):
        #     b_ket = basis_ancilla[b, :].reshape([2 ** n_ancilla, 1])
        #     I_a_b_ket = np.kron(I_system_operator, b_ket)
        #     I_a_b_bra = I_a_b_ket.transpose().conj()
        #
        #     term = I_a_b_bra.dot(projected_density_matrix.dot(I_a_b_ket))
        #     reduced_dm += term

        if not np.isclose(np.trace(reduced_dm), 1):
            raise ValueError('partial density matrix is not normalised properly {}'.format(np.trace(reduced_dm)))

        return reduced_dm

    def Get_standard_ket(self):
        # simulator = cirq.Simulator()
        # output_ket = simulator.compute_amplitudes(self.ansatz_circuit,
        #                                       bitstrings=[i for i in range(2 ** len(self.ansatz_circuit.all_qubits()))])
        #
        input_state = [self.zero_state for _ in range(self.N_system_qubits)]
        input_ket = reduce(kron, input_state)
        circuit_matrix = self.ansatz_circuit.unitary()

        ansatz_state_ket = circuit_matrix.dot(input_ket.todense())

        if not np.isclose(sum([np.abs(i)**2 for i in ansatz_state_ket]), 1):
            raise ValueError('output ket is not normalised properly {}'.format(sum([np.abs(i)**2 for i in ansatz_state_ket])))

        return np.array(ansatz_state_ket) #.reshape([(2 ** len(self.ansatz_circuit.all_qubits())), 1])

    def Get_pauli_matrix(self, PauliOp):
        list_Q_nos, list_P_strs = list(zip(*[Paulistrs for Paulistrs, const in PauliOp.terms.items()][0]))

        list_of_ops = []
        # list_of_ops_print=[]
        for i in range(self.N_system_qubits):
            if i in list_Q_nos:
                index = list_Q_nos.index(i)
                list_of_ops.append(self.pauliDict[list_P_strs[index]])
                # list_of_ops_print.append('{}{}'.format(list_P_strs[index], i))
            else:
                list_of_ops.append(self.pauliDict['I'])
                # list_of_ops_print.append('I{}'.format(i))

        # print(list_of_ops_print, PauliOp)
        matrix = reduce(kron, list_of_ops)

        return matrix

    def Calc_Energy(self, check_LCU_reduction=False):
        # from openfermion.transforms import get_sparse_operator

        E_list = []
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

                ### checking ancilla line!
                N_ancilla_qubits = int(np.ceil(np.log2(len(ancilla_amplitudes))))
                # ancilla_obj = prepare_arb_state(ancilla_amplitudes, self.N_system_qubits) # perfect gate (but cannot decompose)
                ancilla_obj = prepare_arb_state_one_two_qubit_gates(ancilla_amplitudes, self.N_system_qubits) #<-- single and two qubit gate construction!
                ancilla_circ = ancilla_obj.Get_state_prep_Circuit()

                simulator = cirq.Simulator()
                output_ket = simulator.compute_amplitudes(ancilla_circ,
                                                      bitstrings=[i for i in range(2 ** N_ancilla_qubits)])
                # print(ancilla_amplitudes)
                # print(output_ket[:len(ancilla_amplitudes)])
                # print(np.allclose(ancilla_amplitudes, output_ket[:len(ancilla_amplitudes)]))
                # print('###########')
                if not np.allclose(ancilla_amplitudes, output_ket[:len(ancilla_amplitudes)]):
                    print('MISTAKE ON ANCILLA LINE!!!!!!!!!!')



                # gives R|ψ〉
                Q_circuit = Full_Ansatz_and_Quantum_R_circuit(Pn, R_corrected_Op_list, R_corr_list, ancilla_amplitudes,
                                           self.N_system_qubits, self.ansatz_circuit)

                # print(Q_circuit.to_text_diagram(transpose=True))
                # print('')
                # print('##')

                partial_density_matrix = self.Get_parital_system_density_matrix(Q_circuit)

                H_sub_term_matrix = self.Get_pauli_matrix(Pn)
                # H_sub_term_matrix = get_sparse_operator(Pn, n_qubits=self.N_system_qubits)
                # # E=〈ψ | H | ψ〉= ∑_j  αj〈ψA | R† Pn R | ψA〉 #### where RQR = Pn

                # E= Tr(Pn rho)
                energy = np.trace(partial_density_matrix.dot(H_sub_term_matrix.todense()))
                # energy = np.trace(H_sub_term_matrix.dot(partial_density_matrix))
                E_list.append(energy * gamma_l)

            else:
                single_PauliOp = self.anti_commuting_sets[set_key][0]
                if list(single_PauliOp.terms.keys())[0] == ():
                    E_list.append(list(single_PauliOp.terms.values())[0])
                else:
                    ansatz_state_ket = self.Get_standard_ket()
                    ansatz_state_bra = ansatz_state_ket.transpose().conj()
                    # H_sub_term_matrix = get_sparse_operator(single_PauliOp, n_qubits=self.N_system_qubits)

                    # E=〈ψ | H | ψ〉= ∑_j  αj〈ψ | Pj | ψ〉
                    H_sub_term_matrix = self.Get_pauli_matrix(single_PauliOp)
                    # H_sub_term_matrix = get_sparse_operator(single_PauliOp, n_qubits=self.N_system_qubits)
                    energy = ansatz_state_bra.dot(H_sub_term_matrix.todense().dot(ansatz_state_ket))
                    E_list.append(energy.item(0) * list(single_PauliOp.terms.values())[0])

        return sum(E_list)

    def Get_wavefunction_of_ansatz_state(self, sig_figs=3):
        return Get_wavefunction(self.ansatz_circuit, sig_figs=sig_figs)

def Full_Q_Circuit_single_and_two_qubit_gates(Pn, R_corrected_Op_list, R_correction_list, ancilla_amplitudes, N_system_qubits, ansatz_circ):

    """
    class to generate full cirq circuit that prepares a quantum state (ansatz) performs R operation as LCU and finally
    measures both the system and ancilla qubit lines in the Z basis.

    Args:
        Pn (QubitOperator): Qubit operator to measure
        R_corrected_Op_list (list): list of LCU qubit Operators
        R_corr_list (list): list of correction phases for LCU operators
        ancilla_amplitudes (list): list of ancilla amplitudes
        N_system_qubits (int): Number of system qubits / number of spin orbitals
        ansatz_circ (cirq.Circuit): cirq ansatz quantum circuit

    Returns
        A cirq circuit object to be used by cirq.Circuit

    example

    Pn= QubitOperator('Z3', 1)
    R_corrected_Op_list = [0.30200159443367586 [], 0.9533074199645766 [Y0 X1 X2 X3]]
    R_corr_list = [1, (-0-1j)]
    ancilla_amplitudes = [0.30200159443367586, 0.9533074199645766]
    N_system_qubits = 4
    ansatz_circ= Q_circuit

    output:

0: ───X─────────────────Rx(0.5π)───@─────────────────────────────────@───Rx(-0.5π)───I──────Y0──────────────────────────
                                   │                                 │               │      │
1: ───X─────────────────H──────────X───@─────────────────────────@───X───H───────────I──────X1──────────────────────────
                                       │                         │                   │      │
2: ───H────────────────────────────────X───@─────────────────@───X───H───────────────I──────X2──────────────────────────
                                           │                 │                       │      │
3: ───H────────────────────────────────────X───Rz(-1.947π)───X───H───────────────────1*I3───(-0-1j)*X3────────────────M─
                                                                                     │      │                         │
4: ─── U = 1.264 rad ────────────────────────────────────────────────────────────────(0)────@─── ─── U = 1.264 rad ───M─

    """

    ancilla_obj = prepare_arb_state_one_two_qubit_gates(ancilla_amplitudes, N_system_qubits)
    ancilla_circ = ancilla_obj.Get_state_prep_Circuit()
    N_ancilla_qubits = int(np.ceil(np.log2(len(ancilla_amplitudes))))

    R_circ_obj = LCU_R_gate(N_ancilla_qubits, N_system_qubits, R_corrected_Op_list, R_correction_list, Pn)
    R_circ_circ = cirq.Circuit(
        cirq.decompose((R_circ_obj(*cirq.LineQubit.range(R_circ_obj.num_qubits())))))

    change_to_Z_basis_obj = Change_PauliWord_measurement_to_Z_basis(Pn)
    change_to_Z_basis_circ = cirq.Circuit(
        cirq.decompose_once((change_to_Z_basis_obj(*cirq.LineQubit.range(change_to_Z_basis_obj.num_qubits())))))

    measure_obj = Measure_system_and_ancilla(Pn, N_ancilla_qubits, N_system_qubits)

    measure_obj_circ = cirq.Circuit(
        cirq.decompose_once((measure_obj(*cirq.LineQubit.range(measure_obj.num_qubits())))))

    full_Q_circ = cirq.Circuit([
        *ansatz_circ.all_operations(),
        *ancilla_circ.all_operations(),
        *R_circ_circ.all_operations(),
        *list(ancilla_circ.all_operations())[::-1],
        *change_to_Z_basis_circ.all_operations(),
        *measure_obj_circ
    ])
    return full_Q_circ