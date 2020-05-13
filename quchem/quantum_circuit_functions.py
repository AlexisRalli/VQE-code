import cirq
import numpy as np


##### For Ansatze ####

class State_Prep(cirq.Gate):
    """

    Class to generate cirq circuit that creates input state, which is
    defined in occupation number basis (canonical orbitals).

    Args:
        initial_state (list): List in occupation number basis... e.g. |0011>  =  [0,0,1,1]

    Attributes:
        state (list): List in occupation number basis... e.g. |0011>  =  [0,0,1,1]

    """
    def __init__(self, initial_state):

        self.state = np.asarray(initial_state, dtype=int)


    def _decompose_(self, qubits):
        """
        Args:
            qubits (cirq.line.line_qubit.LineQubit): cirq qubits (given to decompose method by cirq)

        Raises:
            ValueError: State not in occupation number basis

        Yields:
            cirq.circuits.circuit.Circuit: cirq circuit generator!
        """

        #for i in range(len(self.state)):
        for qubit_index, qubit_state in enumerate(self.state):
            if qubit_state == 1:
                yield cirq.X(qubits[qubit_index])

            elif qubit_state == 0:
                yield cirq.I(qubits[qubit_index])
            else:
                raise ValueError('initial state not in correct format... qubit {} has value {} ' \
                                 '[instead of 0 or 1]'.format(qubit_index, qubit_state))

    def num_qubits(self):
        return len(self.state)

    def _circuit_diagram_info_(self, args):
        state_prep_list = []
        for qubit_index, qubit_state in enumerate(self.state):
            if qubit_state == 1:
                state_prep_list.append('state_prep: |1> ')
            elif qubit_state == 0:
                state_prep_list.append('state_prep: |0> ')
            else:
                raise ValueError('initial state not in correct format... qubit {} has value {} ' \
                                 '[instead of 0 or 1]'.format(qubit_index, qubit_state))
        return state_prep_list


    def Return_circuit_as_list(self):
        circuit_list=[]
        for i in range(len(self.state)):
            state = self.state[i]
            qubitNo = i
            if state == 1:
                circuit_list.append(cirq.X(qubitNo))
        return circuit_list

##
class Change_of_Basis_initial(cirq.Gate):
    """
    Class to generate cirq circuit as gate... which performs a change of basis for in order to
    perform PauliWord as a Z terms only for: e^(cofactor * theta * PauliWord_Z_ONLY)

    e.g.: QubitOperator('X0 Z2 Y3', 0.5j)
    gives:
            0: ───H──────────

            3: ───Rx(0.5π)───

    Args:
        PauliWord_QubitOp (openfermion.QubitOperator): QubitOperator of PauliWord

    Returns
        A cirq circuit object to be used by cirq.Circuit

    """

    def __init__(self, PauliWord_QubitOp):

        self.PauliWord = list(*PauliWord_QubitOp.terms.keys())

    def _decompose_(self, qubits):

        for qubitNo, PauliStr in (self.PauliWord):
            if PauliStr == 'X':
                yield cirq.H(qubits[qubitNo])
            elif PauliStr == 'Y':
                yield cirq.rx(np.pi / 2)(qubits[qubitNo])
            elif PauliStr == 'Z' or 'I':
                continue
            else:
                raise ValueError("Qubit Operation: {} is NOT a Pauli operation".format(PauliStr))

    def _circuit_diagram_info_(self, args):
        string_list = []
        for _ in range(self.num_qubits()):
            string_list.append('Changing to Z basis circuit')
        return string_list

    def num_qubits(self):
        qubit_list, _ = zip(*(self.PauliWord))
        control_qubit = max(qubit_list)
        return control_qubit + 1  # index from 0

class Engtangle_initial(cirq.Gate):
    """
    Class to generate cirq circuit as gate... which generates CNOT entangling gates between non Idenity PauliWord
    qubits in order to perform PauliWord as a Z terms only for: e^(cofactor * theta * PauliWord_Z_ONLY)

    e.g.: QubitOperator('X0 Z2 Y3', 0.5j)
        gives :
                0: ───@───────
                      │
                2: ───X───@───
                          │
                3: ───────X───

    Args:
        PauliWord_QubitOp (openfermion.QubitOperator): QubitOperator of PauliWord

    Returns
        A cirq circuit object to be used by cirq.Circuit

    """

    def __init__(self, PauliWord_QubitOp):

        self.PauliWord = list(*PauliWord_QubitOp.terms.keys())

    def _decompose_(self, qubits):
        qubit_list, _  = zip(*self.PauliWord)
        control_qubit = max(qubit_list)
        for index, (qubitNo, PauliStr) in enumerate(self.PauliWord):
            if qubitNo < control_qubit:
                qubitNo_NEXT = self.PauliWord[index+1][0]
                yield cirq.CNOT.on(qubits[qubitNo], qubits[qubitNo_NEXT])

    def _circuit_diagram_info_(self, args):
        string_list=[]
        for _ in range(self.num_qubits()):
                string_list.append('Entangling circuit')
        return string_list

    def num_qubits(self):
        qubit_list, _ = zip(*(self.PauliWord))
        control_qubit = max(qubit_list)
        return control_qubit+1 #index from 0

class PauliWord_exponential_rotation(cirq.Gate):
    """
    Class to generate cirq circuit as gate... which generates rotationional gate to perform
    perform PauliWord as a Z terms only for: e^(cofactor * theta * PauliWord_Z_ONLY)

    e.g.: QubitOperator('X0 Z2 Y3', -0.125j) and theta= pi
        gives :
                3: ───Rz(0.25π)───

        NOTE have pi *0.125 * 2 = 0.25pi
        NEED times 2 for definition of: R_z(theta) = e^(Z*theta/2)

    Args:
        PauliWord_QubitOp (openfermion.QubitOperator): QubitOperator of PauliWord
        theta (float): angle to rotate

    Returns
        A cirq circuit object to be used by cirq.Circuit

    """

    def __init__(self, PauliWord_QubitOp, theta):

        self.PauliWord = list(*PauliWord_QubitOp.terms.keys())
        self.theta = theta
        self.cofactor = list(PauliWord_QubitOp.terms.values())[0]

    def _decompose_(self, qubits):

        qubit_list, _ = zip(*self.PauliWord)
        control_qubit = max(qubit_list)

        if self.cofactor.imag < 0:
            yield cirq.rz(2 * self.theta * np.abs(self.cofactor.imag)).on(qubits[control_qubit])
        else:
            # times angle by negative one to get implementation
            yield cirq.rz(2 * self.theta * np.abs(self.cofactor.imag) * -1).on(qubits[control_qubit])

    def _circuit_diagram_info_(self, args):
        string_list = []
        for _ in range(self.num_qubits()):
            string_list.append('performing Z rotation circuit')
        return string_list

    def num_qubits(self):
        qubit_list, _ = zip(*(self.PauliWord))
        control_qubit = max(qubit_list)
        return control_qubit + 1  # index from 0

class Engtangle_final(cirq.Gate):
    """
    Class to generate cirq circuit as gate... which generates CNOT entangling gates between non Idenity PauliWord
    qubits in order to perform PauliWord as a Z terms only for: e^(cofactor * theta * PauliWord_Z_ONLY)

    e.g.: QubitOperator('X0 Z2 Y3', 0.5)
        gives :
                0: ───────@───
                          │
                2: ───@───X───
                      │
                3: ───X───────

    Args:
        PauliWord_QubitOp (openfermion.QubitOperator): QubitOperator of PauliWord

    Returns
        A cirq circuit object to be used by cirq.Circuit

    """

    def __init__(self, PauliWord_QubitOp):

        self.PauliWord = list(*PauliWord_QubitOp.terms.keys())

    def _decompose_(self, qubits):

        qubit_list, _ = zip(*self.PauliWord)
        min_qubit = min(qubit_list)
        for index, (qubitNo, PauliStr) in enumerate(self.PauliWord[::-1]):
            if qubitNo > min_qubit:
                qubitNo_next = self.PauliWord[::-1][index + 1][0]
                yield cirq.CNOT.on(qubits[qubitNo_next],qubits[qubitNo])

    def _circuit_diagram_info_(self, args):
        string_list = []
        for _ in range(self.num_qubits()):
            string_list.append('Entangling circuit')
        return string_list

    def num_qubits(self):
        qubit_list, _ = zip(*(self.PauliWord))
        control_qubit = max(qubit_list)
        return control_qubit + 1  # index from 0

class Change_of_Basis_final(cirq.Gate):
    """
    Class to generate cirq circuit as gate... which generates CNOT entangling gates between non Idenity PauliWord
    qubits in order to perform PauliWord: e^(cofactor * theta * PauliWord)

    e.g.: QubitOperator('X0 Z2 Y3', 0.5)
        gives :
                0: ───H───────────

                3: ───Rx(-0.5π)───

    Args:
        PauliWord_QubitOp (openfermion.QubitOperator): QubitOperator of PauliWord

    Returns
        A cirq circuit object to be used by cirq.Circuit

    """

    def __init__(self, PauliWord_QubitOp):

        self.PauliWord = list(*PauliWord_QubitOp.terms.keys())

    def _decompose_(self, qubits):

        for qubitNo, PauliStr in (self.PauliWord):
            if PauliStr == 'X':
                yield cirq.H(qubits[qubitNo])
            elif PauliStr == 'Y':
                yield cirq.rx(-np.pi / 2)(qubits[qubitNo])
            elif PauliStr == 'Z' or 'I':
                continue
            else:
                raise ValueError("Qubit Operation: {} is NOT a Pauli operation".format(PauliStr))

    def _circuit_diagram_info_(self, args):
        string_list = []
        for _ in range(self.num_qubits()):
            string_list.append('Changing to old basis circuit')
        return string_list

    def num_qubits(self):
        qubit_list, _ = zip(*(self.PauliWord))
        control_qubit = max(qubit_list)
        return control_qubit + 1  # index from 0

class full_exponentiated_PauliWord_circuit(cirq.Gate):
    """
    Class to generate cirq circuit as gate performing : e^(cofactor * theta * PauliWord_Z_ONLY)

    e.g.: QubitOperator('X0 Z2 Y3', 0.25j) and theta= pi
        gives :
                0: ───H──────────@───────────────────────X───────────H───
                                 │                       │
                2: ──────────────X───@───────────────X───@───────────────
                                     │               │
                3: ───Rx(0.5π)───────X───Rz(-0.5π)───@───Rx(-0.5π)───────

    Args:
        PauliWord_QubitOp (openfermion.QubitOperator): QubitOperator of PauliWord
        theta (float): angle to rotate

    Returns
        A cirq circuit object to be used by cirq.Circuit

    """

    def __init__(self, PauliWord_QubitOp, theta):
        self.PauliWord_QubitOp = PauliWord_QubitOp
        self.theta = theta

    def _decompose_(self, qubits):
        Basis_change_initial_circuit = Change_of_Basis_initial(self.PauliWord_QubitOp)
        Ent_initial = Engtangle_initial(self.PauliWord_QubitOp)
        PauliWord_exponential_rotation_circuit = PauliWord_exponential_rotation(self.PauliWord_QubitOp, self.theta)
        Ent_final = Engtangle_final(self.PauliWord_QubitOp)
        Basis_change_final_circuit = Change_of_Basis_final(self.PauliWord_QubitOp)

        basis_change_initial_gen = Basis_change_initial_circuit._decompose_(qubits)
        Ent_initial_gen = Ent_initial._decompose_(qubits)
        PauliWord_exponential_rotation_circuit_gen = PauliWord_exponential_rotation_circuit._decompose_(qubits)
        Ent_final_gen = Ent_final._decompose_(qubits)
        basis_change_final_gen = Basis_change_final_circuit._decompose_(qubits)

        list_generators = [basis_change_initial_gen, Ent_initial_gen, PauliWord_exponential_rotation_circuit_gen,
                           Ent_final_gen,
                           basis_change_final_gen]
        yield list_generators

    def _circuit_diagram_info_(self, args):
        string_list = []
        for _ in range(self.num_qubits()):
            string_list.append(
                'exponentiated pauli: {} with angle: {}'.format(self.PauliWord_QubitOp, np.round(self.theta, 3)))
        return string_list

    def num_qubits(self):
        qubit_list, _ = zip(*(list(*self.PauliWord_QubitOp.terms.keys())))
        control_qubit = max(qubit_list)
        return control_qubit + 1  # index from 0


####### For Hamiltonian ####

class Change_PauliWord_measurement_to_Z_basis(cirq.Gate):
    """
    Class to generate cirq circuit as gate... which generates CNOT entangling gates between non Idenity PauliWord
    qubits in order to perform PauliWord: e^(cofactor * theta * PauliWord)

    e.g.: QubitOperator('X0 Z2 Y3', 0.5)
        gives :
                0: ───H───────────

                3: ───Rx(-0.5π)───list_of_Pn_qNos

    Args:
        PauliWord_QubitOp (openfermion.QubitOperator): QubitOperator of PauliWord

    Returns
        A cirq circuit object to be used by cirq.Circuit

    """

    def __init__(self, PauliWord_QubitOp):

        self.PauliWord = list(*PauliWord_QubitOp.terms.keys())

    def _decompose_(self, qubits):

        for qubitNo, PauliStr in (self.PauliWord):
            if PauliStr == 'X':
                # yield cirq.H(qubits[qubitNo])
                yield cirq.ry(-np.pi / 2)(qubits[qubitNo])
            elif PauliStr == 'Y':
                yield cirq.rx(np.pi / 2)(qubits[qubitNo])
            elif PauliStr == 'Z' or 'I':
                continue
            else:
                raise ValueError("Qubit Operation: {} is NOT a Pauli operation".format(PauliStr))

    def _circuit_diagram_info_(self, args):
        string_list = []
        for _ in range(self.num_qubits()):
            string_list.append('Changing pauliword to Z basis')
        return string_list

    def num_qubits(self):
        qubit_list, _ = zip(*(self.PauliWord))
        control_qubit = max(qubit_list)
        return control_qubit + 1  # index from 0

class Measure_PauliWord(cirq.Gate):
    """
    Class to generate cirq circuit as gate that measures PauliWord in Z BASIS (all non I terms).

    e.g.: QubitOperator('X0 Z2 Y3', 0.5)
        gives :
                0: ───M───
                      │
                2: ───M───
                      │
                3: ───M───
    Args:
        PauliWord_QubitOp (openfermion.QubitOperator): QubitOperator of PauliWord

    Returns
        A cirq circuit object to be used by cirq.Circuit

    """

    def __init__(self, PauliWord_QubitOp):

        self.PauliWord = list(*PauliWord_QubitOp.terms.keys())

    def _decompose_(self, qubits):

        # list of qubits to measure!
        qubit_list, _ = zip(*(self.PauliWord))

        qubits_to_measure = (qubits[q_No] for q_No in qubit_list)
        if qubit_list != []:
            yield cirq.measure(*qubits_to_measure)
        else:
            return None

    def _circuit_diagram_info_(self, args):
        string_list = []
        for _ in range(self.num_qubits()):
            string_list.append('measuring pauliword')
        return string_list

    def num_qubits(self):
        qubit_list, _ = zip(*(self.PauliWord))
        control_qubit = max(qubit_list)
        return control_qubit + 1  # index from 0

class change_pauliword_to_Z_basis_then_measure(cirq.Gate):
    """
    Class to generate cirq circuit as gate that changes basis and measures PauliWord in Z BASIS (all non I terms).

    e.g.: QubitOperator('X0 Z2 Y3', 0.5)
        gives :
                0: ───H───────────M───
                                  │
                2: ───────────────M───
                                  │
                3: ───Rx(-0.5π)───M───

    Args:
        PauliWord_QubitOp (openfermion.QubitOperator): QubitOperator of PauliWord

    Returns
        A cirq circuit object to be used by cirq.Circuit

    """

    def __init__(self, PauliWord_QubitOp):
        self.PauliWord_QubitOp = PauliWord_QubitOp

    def _decompose_(self, qubits):
        change_basis_to_measure_circuit = Change_PauliWord_measurement_to_Z_basis(self.PauliWord_QubitOp)
        Measure_circuit = Measure_PauliWord(self.PauliWord_QubitOp)

        change_basis_to_measure_circuit_gen = change_basis_to_measure_circuit._decompose_(qubits)
        Measure_circuit_gen = Measure_circuit._decompose_(qubits)

        list_gen = [change_basis_to_measure_circuit_gen, Measure_circuit_gen]
        yield list_gen

    def _circuit_diagram_info_(self, args):
        string_list = []
        for _ in range(self.num_qubits()):
            string_list.append('measuring pauliword: {}'.format(self.PauliWord_QubitOp))
        return string_list

    def num_qubits(self):
        qubit_list, _ = zip(*(list(*self.PauliWord_QubitOp.terms.keys())))
        control_qubit = max(qubit_list)
        return control_qubit + 1  # index from 0

def Generate_Full_Q_Circuit(Full_Ansatz_Q_Circuit, PauliWord_QubitOp):
    """
     Function that appends Ansatz Quantum Circuit to Pauli perform and measure circuit instance.

    Args:
        Full_Ansatz_Q_Circuit (cirq.circuits.circuit.Circuit): Full cirq Ansatz Q circuit
        PauliWord_QubitOp (openfermion.QubitOperator): QubitOperator of PauliWord

    Returns:
        full_circuit (cirq.circuits.circuit.Circuit): Full cirq VQE circuit

    """

    measure_PauliString_in_Z_basis = change_pauliword_to_Z_basis_then_measure(PauliWord_QubitOp)
    measure_PauliString_in_Z_basis_Q_circ = cirq.Circuit(cirq.decompose_once(
        (measure_PauliString_in_Z_basis(*cirq.LineQubit.range(measure_PauliString_in_Z_basis.num_qubits())))))
    full_circuit = cirq.Circuit(
       [
           Full_Ansatz_Q_Circuit.all_operations(),
           *measure_PauliString_in_Z_basis_Q_circ.all_operations(),
       ]
    )
    return full_circuit

def Generate_Full_Q_Circuit_of_Molecular_Hamiltonian(Full_Ansatz_Q_Circuit, QubitOperator_Hamiltonian):
    """
     Function that appends Ansatz Quantum Circuit to Pauli perform and measure circuit instance.

    Args:
        Full_Ansatz_Q_Circuit (cirq.circuits.circuit.Circuit): Full cirq Ansatz Q circuit
        PauliWord_str_list (list): List of tuples containing PauliWord to perform and constant -> (PauliWord, constant)
        n_qubits (int): number of qubits

    Returns:
        dic_holder (dict): Returns a dictionary of each quantum circuit, with cofactor, PauliWord and cirq Q Circuit

    """
    dic_holder = {}
    for index, PauliWord_QubitOp in enumerate(QubitOperator_Hamiltonian):
        for QubitOp_str, const in PauliWord_QubitOp.terms.items():
            temp_d = {}
            if QubitOp_str:
                temp_d['circuit'] = Generate_Full_Q_Circuit(Full_Ansatz_Q_Circuit, PauliWord_QubitOp)
                temp_d['PauliWord'] = PauliWord_QubitOp
            else:
                temp_d['circuit'] = None
                temp_d['PauliWord'] = PauliWord_QubitOp

            dic_holder[index] = temp_d

    return dic_holder


#### Prepare arb state ###

def Get_state_as_str(n_qubits, qubit_state_int):
    """
    converts qubit state int into binary form.

    Args:
        n_qubits (int): Number of qubits
        qubit_state_int (int): qubit state as int (NOT BINARY!)
    Returns:
        string of qubit state in binary!

    state = |000> + |001> + |010> + |011> + |100> + |101 > + |110 > + |111>
    state  = |0> +   |1> +   |2> +   |3> +   |4> +   |5 > +   |6 > +   |7>

    n_qubits = 3
    state = 5
    Get_state_as_str(n_qubits, state)
    >> '101'

    """
    bin_str_len = '{' + "0:0{}b".format(n_qubits) + '}'
    return bin_str_len.format(qubit_state_int)

class My_U_Gate(cirq.SingleQubitGate):
    """
    Description

    Args:
        theta (float): angle to rotate by in radians.
        number_control_qubits (int): number of control qubits
    """

    def __init__(self, theta):
        self.theta = theta
    def _unitary_(self):
        Unitary_Matrix = np.array([
                    [np.cos(self.theta), np.sin(self.theta)],
                    [np.sin(self.theta), -1* np.cos(self.theta)]
                ])
        return Unitary_Matrix
    def num_qubits(self):
        return 1

    def _circuit_diagram_info_(self,args):
        # return cirq.CircuitDiagramInfo(
        #     wire_symbols=tuple([*['@' for _ in range(self.num_control_qubits-1)],' U = {} rad '.format(self.theta.__round__(4))]),exponent=1)
        return ' U = {} rad '.format(self.theta.__round__(4))

def Get_control_parameters(num_qubits, Coefficient_list):
    if len(Coefficient_list) != 2 ** num_qubits:
        Coefficient_list = Coefficient_list + [0 for _ in range(2 ** num_qubits - len(Coefficient_list))]
        #raise ValueError('incorrect number of coefficients')

    state_list = [Get_state_as_str(num_qubits, i) for i in range(2 ** num_qubits)]

    alpha_j_dict = {}
    for target_qubit in range(num_qubits - 1):

        number_controls = target_qubit

        if number_controls > 0:
            CONTROL_state_list = [Get_state_as_str(number_controls, i) for i in range(2 ** number_controls)]
        else:
            CONTROL_state_list = ['']

        term_list = []
        for control_state in CONTROL_state_list:
            top_term_str = control_state + '1'
            bottom_term_str = control_state + '0'

            top = 0
            bottom = 0
            for index, state_str in enumerate(state_list):
                if state_str[:target_qubit + 1] == top_term_str:
                    top += Coefficient_list[index] ** 2

                if state_str[:target_qubit + 1] == bottom_term_str:
                    bottom += Coefficient_list[index] ** 2
                else:
                    continue

            if (bottom == 0) and (top == 0):
                angle = 0
            else:
                try:
                    angle = np.arctan(np.sqrt(top / bottom))
                except:
                    raise ValueError('undetermined angle! NEED TO CHECK PROBLEM')

            term_list.append({'control_state': control_state, 'angle': angle})
        alpha_j_dict[target_qubit] = term_list

    ##final rotation ##
    if num_qubits!=1:
        term_list = []
        for index, state_str in enumerate([Get_state_as_str((num_qubits - 1), i) for i in range(2 ** (num_qubits - 1))]):
            control_state_str = state_str

            top_term_str = control_state_str + '1'
            bottom_term_str = control_state_str + '0'

            index_top = state_list.index(top_term_str)
            index_bottom = state_list.index(bottom_term_str)

            top = Coefficient_list[index_top]
            bottom = Coefficient_list[index_bottom]

            if (bottom == 0) and (top == 0):
                angle = 0
            else:
                try:
                    angle = np.arctan(top / bottom)
                except:
                    raise ValueError('undetermined angle! NEED TO CHECK PROBLEM')

            term_list.append({'control_state': control_state_str, 'angle': angle})

        alpha_j_dict[num_qubits - 1] = term_list

        return alpha_j_dict
    else:

        # [np.cos(self.theta), np.sin(self.theta)],         [1]  =  [a]
        # [np.sin(self.theta), -1 * np.cos(self.theta)]     [0]     [b]
        theta = np.arccos(Coefficient_list[0])
        alpha_j_dict[0] = [{'control_state': '', 'angle': theta}]

        return alpha_j_dict



class State_Prep_Circuit(cirq.Gate):
    """
    Function to build cirq Circuit that will make an arbitrary state!

    e.g.:
    {
         0: [{'control_state': None, 'angle': 0.7853981633974483}],
         1: [{'control_state': '0', 'angle': 0.7853981633974483},
          {'control_state': '1', 'angle': 0.7853981633974483}]
      }

gives :

0: ── U = 0.51 rad ──(0)─────────────@──────────────(0)────────────(0)──────────────@────────────────@────────────────
                     │               │              │              │                │                │
1: ────────────────── U = 0.91 rad ── U = 0.93 rad ─(0)────────────@────────────────(0)──────────────@────────────────
                                                    │              │                │                │
2: ───────────────────────────────────────────────── U = 0.30 rad ─ U = 0.59 rad ─── U = 0.72 rad ─── U = 0.71 rad ───

    Args:
        circuit_param_dict (dict): A Dictionary of Tuples (qubit, control_val(int)) value is angle

    Returns
        A cirq circuit object to be used by cirq.Circuit.from_ops to generate arbitrary state

    """

    def __init__(self, circuit_param_dict, N_system_qubits=0):

        self.circuit_param_dict = circuit_param_dict
        self.N_system_qubits = N_system_qubits

    def _decompose_(self, qubits):

        for qubit in self.circuit_param_dict:

            for term in self.circuit_param_dict[qubit]:
                if term['control_state']:
                    control_values = [int(bit) for bit in term['control_state']]
                    num_controls = len(control_values)
                    theta = term['angle']

                    if theta == 0:
                        yield cirq.I.on(cirq.LineQubit(qubit+self.N_system_qubits))
                    else:
                        U_single_qubit = My_U_Gate(theta)
                        qubit_list = cirq.LineQubit.range(self.N_system_qubits, self.N_system_qubits+1 + num_controls)
                        yield U_single_qubit.controlled(num_controls=num_controls, control_values=control_values).on(
                            *qubit_list)
                #                     U_single_qubit = My_U_Gate(theta)
                #                     qubit_list = cirq.LineQubit.range(0,1+num_controls)
                #                     yield U_single_qubit.controlled(num_controls=num_controls, control_values=control_values).on(*qubit_list)
                else:
                    theta = term['angle']
                    if theta == 0:
                        yield cirq.I.on(cirq.LineQubit(qubit+self.N_system_qubits))
                    else:
                        yield My_U_Gate(theta).on(cirq.LineQubit(qubit+self.N_system_qubits))

    #                     theta = term['angle']
    #                     yield My_U_Gate(theta).on(cirq.LineQubit(qubit))

    def _circuit_diagram_info_(self, args):

        max_qubit = max(self.circuit_param_dict.keys())
        string_list = []
        for i in range(max_qubit):
            string_list.append('state prep circuit')
        return string_list

    def num_qubits(self):
        return max(self.circuit_param_dict.keys())

class prepare_arb_state():
    def __init__(self, Coefficient_list, N_System_qubits):
        self.Coefficient_list = Coefficient_list
        self.N_System_qubits = N_System_qubits
    def _Get_control_parameters_dict(self):
        return Get_control_parameters(self.Get_max_no_ancilla_qubits(), self.Coefficient_list)

    def Get_state_prep_Circuit(self):
        circ_obj = State_Prep_Circuit(self._Get_control_parameters_dict(), self.N_System_qubits)
        circuit = (
            cirq.Circuit(cirq.decompose_once((circ_obj(*cirq.LineQubit.range(self.N_System_qubits, self.N_System_qubits+circ_obj.num_qubits()))))))
        return circuit

    def Get_max_no_ancilla_qubits(self):
        return int(np.ceil(np.log2(len(self.Coefficient_list))))  # note round up with np.ceil

    def get_wave_function(self, sig_figs=3):
        circuit = self.Get_state_prep_Circuit()
        simulator = cirq.Simulator()
        result = simulator.compute_amplitudes(circuit, bitstrings=[i for i in range(2 ** len(circuit.all_qubits()))])
        result = np.around(result, sig_figs)
        return result.reshape([(2 ** len(circuit.all_qubits())), 1])

