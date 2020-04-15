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
                                               mulitplying_const=1j)  # here we times by 1j due to defintion of X also we divided by B_n (as we only want to multiply by P_n B#NOT B_n P_n)
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

def Get_R_linear_combination(anti_commuting_set, N_index):
    """
    """
    X_set = Get_X_SET(anti_commuting_set, N_index)

    # χ = 𝑖 ∑ 𝛿_𝑘 𝑃_𝑘𝑛
    X_terms = X_set['X_PauliWords']

    # 𝛼 = 𝜙_{𝑛−1}
    alpha = X_set['phi_n_1']

    # 𝑅=cos(𝛼/2)𝟙−𝑖sin(𝛼/2)χ = cos(𝛼/2)𝟙−𝑖sin(𝛼/2)*(𝑖 ∑ 𝛿_𝑘 𝑃_𝑘𝑛)

    # cos(𝛼/2)𝟙 term
    I_term = QubitOperator('', np.cos(alpha / 2))
    R_linear_comb_list = [I_term]

    # −𝑖 sin(𝛼/2) * (𝑖 ∑ 𝛿_𝑘 𝑃_𝑘𝑛) terms!
    sin_term = np.sin(alpha / 2) * -1j
    for qubitOp_P_kn in X_terms:
        for P_kn_word, constant in qubitOp_P_kn.terms.items():
            new_constant = sin_term * constant
            R_linear_comb_list.append(QubitOperator(P_kn_word, new_constant))

    if not np.isclose(sum(np.absolute(list(qubitOp.terms.values())[0]) ** 2 for qubitOp in R_linear_comb_list), 1):
        raise ValueError('normalisation of R operator incorrect: {}'.format(
            sum(list(qubitOp.terms.values())[0] ** 2 for qubitOp in R_linear_comb_list)))

    return R_linear_comb_list, X_set['P_n'], X_set['gamma_l']  # , X_set['Omega_l']

def absorb_complex_phases(R_linear_comb_list):
    R_linear_comb_corrected_phase = []
    R_linear_comb_correction_values = []
    ancilla_amplitudes = []

    l2_norm = sum([abs(const) ** 2 for qubitOp in R_linear_comb_list for PauliWord, const in qubitOp.terms.items()])
    if l2_norm > 1:
        raise ValueError('l2_norm means correct amps not obtained')

    for qubitOp in R_linear_comb_list:
        for pauliword, const in qubitOp.terms.items():
            if (isinstance(const, complex)) and (const.imag < 0):
                R_linear_comb_corrected_phase.append(QubitOperator(pauliword, np.absolute(const)))
                R_linear_comb_correction_values.append(-1j)
                ancilla_amplitudes.append(np.sqrt(const.imag ** 2))
            elif (isinstance(const, complex)) and (const.imag != 0):
                R_linear_comb_corrected_phase.append(QubitOperator(pauliword, np.absolute(const)))
                R_linear_comb_correction_values.append(1j)
                ancilla_amplitudes.append(np.sqrt(const.imag ** 2))
            elif const < 0:
                R_linear_comb_corrected_phase.append(QubitOperator(pauliword, np.absolute(const)))
                R_linear_comb_correction_values.append(-1)
                ancilla_amplitudes.append(np.sqrt(const ** 2))
            else:
                R_linear_comb_corrected_phase.append(QubitOperator(pauliword, np.absolute(const)))
                R_linear_comb_correction_values.append(1)
                ancilla_amplitudes.append(np.sqrt(const ** 2))
    return R_linear_comb_corrected_phase, R_linear_comb_correction_values, ancilla_amplitudes

class Perform_modified_Pauligate(cirq.SingleQubitGate):
    """


    The function finds eigenvalue of operator and THEN gives corresponding operator to change to Z basis for measurement!

    Args:
        LCU_PauliWord_and_cofactor (tuple): Tuple of PauliWord (str) and constant (complex) ... (PauliWord, constant)
        LCU_correction_value (complex):

    Returns
        A cirq circuit object to be used by cirq.Circuit

    """

    def __init__(self, PauliStr, correction_value):

        self.PauliStr = PauliStr
        self.correction_value = correction_value

    def _unitary_(self):

        from scipy.linalg import eig

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
        return 'PauliMod : {} gate times {}'.format(self.PauliStr, self.correction_value)

class Perform_Modified_PauliWord(cirq.Gate):
    """
    Class to generate cirq circuit as gate that performs a modified PauliWord

    Args:
        LCU_PauliWord_and_cofactor (tuple): Tuple of PauliWord (str) and constant (complex) ... (PauliWord, constant)
        LCU_correction_value (complex):

    Returns
        A cirq circuit object to be used by cirq.Circuit

    e.g.
        test_case = ('Y0 Z1 X2 I3', 0.00070859248123462)
        correction_val = (-0 - 1j)
        dag = False

        P_circ_mod = Perform_Modified_PauliWord(test_case, correction_val, dag)
        print(cirq.Circuit(
        cirq.decompose_once((P_circ_mod(*cirq.LineQubit.range(P_circ_mod.num_qubits()))))))
        >>
                0: ───change to Z basis for modified PauliMod : Y times -1j───
                1: ───change to Z basis for modified PauliMod : Z times -1j───
                2: ───change to Z basis for modified PauliMod : X times -1j───
                3: ───change to Z basis for modified PauliMod : I times -1j───

    """

    def __init__(self, PauliQubitOp, correction_val):

        self.PauliQubitOp = PauliQubitOp
        self.correction_val = correction_val

    def _decompose_(self, qubits):

        if list(self.PauliQubitOp.terms.keys())[0] == ():
            # identity operations
            pass
        else:
            qubitNos_list, P_strs_list = zip(*list(self.PauliQubitOp.terms.keys())[0])

            for index, P_str in enumerate(P_strs_list):
                yield Perform_modified_Pauligate(P_str, self.correction_val).on(qubits[qubitNos_list[index]])

    def _circuit_diagram_info_(self, args):
        string_list = []
        for _ in range(self.num_qubits()):
            string_list.append('modified P_Word gate')
        return string_list

    def num_qubits(self):
        if list(self.PauliQubitOp.terms.keys())[0] == ():
            # identity operations
            return 0
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

    def __init__(self, No_control_qubits, No_system_qubits, R_corrected_Op_list, R_correction_list):

        self.No_control_qubits = No_control_qubits
        self.No_system_qubits = No_system_qubits
        self.R_corrected_Op_list = R_corrected_Op_list
        self.R_correction_list = R_correction_list

    def _decompose_(self, qubits):

        for control_state_index, R_qubitOp_corrected in enumerate(self.R_corrected_Op_list):

            if list(R_qubitOp_corrected.terms.keys())[0] != ():
                control_str = Get_state_as_str(self.No_control_qubits, control_state_index)
                control_values = [int(bit) for bit in control_str]

                #                     qubit_list = cirq.LineQubit.range(self.No_system_qubits, self.No_system_qubits + self.No_control_qubits)  # note control qubits first!
                qubit_list = cirq.LineQubit.range(self.No_system_qubits, self.No_system_qubits + self.No_control_qubits) \
                             + cirq.LineQubit.range(self.No_system_qubits)  # note control qubits first!

                mod_p_word_gate = Perform_Modified_PauliWord(R_qubitOp_corrected,
                                                             self.R_correction_list[control_state_index])

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

    e.g.: PauliWord_and_cofactor = ('X0 Y1 Z2 I3 Y4', -0.28527408634774526j)
          n_ancilla_qubits = 2

        gives :
                0: ───M───
                      │
                1: ───M───
                      │
                2: ───M───
                      │
                4: ───M───
                      │
                5: ───M───
                      │
                6: ───M───

    Args:
        PauliWord_and_cofactor (tuple): Tuple of PauliWord (str) and constant (complex) ... (PauliWord, constant)
        n_ancilla_qubits (int): Number of ancilla qubits

    Returns
        A cirq circuit object to be used by cirq.Circuit

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

def Full_Q_Circuit(Pn, R_corrected_Op_list, R_correction_list, ancilla_amplitudes, N_system_qubits, Pauli_N,ansatz_circ):
    ancilla_obj = prepare_arb_state(ancilla_amplitudes, N_system_qubits)
    ancilla_circ = ancilla_obj.Get_state_prep_Circuit()

    N_ancilla_qubits = ancilla_obj.Get_max_no_ancilla_qubits()
    ancilla_prep_circ = ancilla_obj.Get_state_prep_Circuit()
    R_circ_obj = LCU_R_gate(N_ancilla_qubits, N_system_qubits, R_corrected_Op_list, R_correction_list)
    R_circ_circ = cirq.Circuit(
        cirq.decompose_once((R_circ_obj(*cirq.LineQubit.range(R_circ_obj.num_qubits())))))

    change_to_Z_basis_obj = Change_PauliWord_measurement_to_Z_basis(Pn)
    change_to_Z_basis_circ = cirq.Circuit(
        cirq.decompose_once((change_to_Z_basis_obj(*cirq.LineQubit.range(change_to_Z_basis_obj.num_qubits())))))

    measure_obj = Measure_system_and_ancilla(Pauli_N, N_ancilla_qubits, N_system_qubits)

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

    qubit_No, PauliStr = zip(*list(*qubitOperator.terms.keys()))

    histogram_string = ','.join([str(i) for i in (qubit_No)] + [str(i) for i in range(N_system_qubits, N_system_qubits + N_ancilla_qubits)])
    return histogram_string

def simulate_probabilistic_Q_circuit(probability_of_success, Quantum_circuit, n_shots):

    simulator = cirq.Simulator()
    raw_result = simulator.run(Quantum_circuit, repetitions=n_shots*int(np.ceil(1/probability_of_success)))
    return raw_result

def Get_binary_dict_project(Quantum_circuit, qubitOperator, n_shots, N_system_qubits, ancilla_amplitudes):
    N_system_terms_measured = len(list(qubitOperator.terms.keys())[0])
    N_ancilla_qubits = int(np.ceil(np.log2(len(ancilla_amplitudes))))
    correct_ancilla_state = np.zeros([N_ancilla_qubits])

    l1_norm = sum(abs(i) for i in ancilla_amplitudes)
    P_success = (1 / l1_norm) ** 2

    n_success_shots = 0
    binary_results_dict = {}
    while n_success_shots != n_shots:
        hist_key = Get_Histogram_key_ancilla_system(qubitOperator, N_system_qubits, N_ancilla_qubits)
        raw_result = simulate_probabilistic_Q_circuit(P_success, Quantum_circuit, n_shots)

        M_results = raw_result.measurements[hist_key]
        for result in M_results:

            #             print('full result: ', result)
            #             print('correct_ancilla_state: ', correct_ancilla_state)
            #             print('aniclla result: ', result[N_system_terms_measured::])

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

            #                 print(binary_results_dict)

            #             else:
            #                 print('fail')

            if n_success_shots == n_shots:
                break
    return binary_results_dict

from quchem.Simulating_Quantum_Circuit import *

class VQE_Experiment_LCU_UP():
    def __init__(self,
                 anti_commting_sets,
                 ansatz_circuit,
                 n_shots,
                 ancilla_amplitudes,
                 N_system_qubits,
                 N_indices_dict=None):

        self.anti_commting_sets = anti_commting_sets
        self.ansatz_circuit = ansatz_circuit
        self.n_shots = n_shots

        self.ancilla_amplitudes = ancilla_amplitudes
        self.N_system_qubits = N_system_qubits
        self.N_indices_dict = N_indices_dict
        self.N_ancilla_qubits = int(np.ceil(np.log2(len(self.ancilla_amplitudes))))

    def Get_Histogram_key_ancilla_system(self, qubitOperator):

        qubit_No, PauliStr = zip(*list(*qubitOperator.terms.keys()))

        histogram_string = ','.join([str(i) for i in (qubit_No)] + [str(i) for i in range(self.N_system_qubits,
                                                                                          self.N_system_qubits + self.N_ancilla_qubits)])
        return histogram_string

    def simulate_probabilistic_Q_circuit(self, probability_of_success, Quantum_circuit):

        simulator = cirq.Simulator()
        raw_result = simulator.run(Quantum_circuit, repetitions=self.n_shots * int(np.ceil(1 / probability_of_success)))
        return raw_result

    def Get_binary_dict_project(self, Quantum_circuit, qubitOperator):

        correct_ancilla_state = np.zeros([self.N_ancilla_qubits])
        N_system_terms_measured = len(list(qubitOperator.terms.keys())[0])

        l1_norm = sum(abs(i) for i in self.ancilla_amplitudes)
        P_success = (1 / l1_norm) ** 2

        n_success_shots = 0
        binary_results_dict = {}
        while n_success_shots != self.n_shots:
            hist_key = self.Get_Histogram_key_ancilla_system(qubitOperator)
            raw_result = self.simulate_probabilistic_Q_circuit(P_success, Quantum_circuit)

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

                #                 print(binary_results_dict)
                if n_success_shots == self.n_shots:
                    break
        return binary_results_dict

    def Calc_Energy(self):

        E_list = []
        for set_key in self.anti_commting_sets:
            if len(self.anti_commting_sets[set_key]) > 1:

                if self.N_indices_dict is None:
                    # chooses Pauli_S to be zero index!
                    R_uncorrected, Pn, gamma_l = Get_R_linear_combination(self.anti_commting_sets[set_key], 0)
                    R_corrected_Op_list, R_corr_list, ancilla_amplitudes = absorb_complex_phases(R_uncorrected)
                else:
                    R_uncorrected, Pn, gamma_l = Get_R_linear_combination(self.anti_commting_sets[set_key],
                                                                          self.N_indices_dict[set_key])
                    R_corrected_Op_list, R_corr_list, ancilla_amplitudes = absorb_complex_phases(R_uncorrected)

                Q_circuit = Full_Q_Circuit(Pn, R_corrected_Op_list,
                                           R_corr_list, ancilla_amplitudes,
                                           self.N_system_qubits,
                                           Pn,
                                           self.ansatz_circuit)

                binary_state_counter = self.Get_binary_dict_project(Q_circuit, Pn)
                exp_result = expectation_value_by_parity(binary_state_counter)
                #                 E_list.append(exp_result*gamma_l)
                E_list.append(exp_result * list(Pn.terms.values())[0] * gamma_l)
                #                 E_list.append(exp_result*list(Pn.terms.values())[0])
                #                 E_list.append(exp_result)
                #                 print(Pn, list(Pn.terms.values())[0])
                print(Pn, gamma_l, exp_result)
            #                 print(np.prod([i**2 for i in R_corr_list]), gamma_l, exp_result)

            else:
                single_PauliOp = self.anti_commting_sets[set_key][0]
                if list(single_PauliOp.terms.keys())[0] == ():
                    E_list.append(list(single_PauliOp.terms.values())[0])
                else:
                    Q_circuit = Generate_Full_Q_Circuit(self.ansatz_circuit, single_PauliOp)
                    hist_key_str = Get_Histogram_key(single_PauliOp)
                    int_state_counter = Simulate_Quantum_Circuit(Q_circuit, self.n_shots, hist_key_str)
                    binary_state_counter = Return_as_binary(int_state_counter, hist_key_str)
                    exp_result = expectation_value_by_parity(binary_state_counter)
                    E_list.append(exp_result * list(single_PauliOp.terms.values())[0])
                    print(single_PauliOp, exp_result * list(single_PauliOp.terms.values())[0])

        #         print(Q_circuit.to_text_diagram(transpose=True))
        return sum(E_list)

    def Get_wavefunction_of_state(self, sig_figs=3):
        return Get_wavefunction(self.ansatz_circuit, sig_figs=sig_figs)
