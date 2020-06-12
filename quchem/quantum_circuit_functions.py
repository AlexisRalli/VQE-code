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

### gate count

def Total_Gate_Count(Q_Circuit, M_gates_included=True, only_one_two_qubit_gates=False):
    """
    Function to count number of gates in a cirq quantum circuit

    note how decomposed quantum circuit matters!
    e.g.
         ──X0────                  ─────────────────────X0──
           │                                            │
         ──Y1────                  ───────────────Y1────────
           │                                      │     │
         ──Y2────           vs     ─────────Y2──────────────
           │                                │     │     │
         ──1j*Y3─                  ──1j*Y3──────────────────
           │                         │      │     │     │
         ──@─────                  ──@──────@─────@─────@───

      one * 5 qubit gate    VS        four * 2 qubit gates

    for example:

    0: ───X──────────────Rx(0.5π)───@──────────────────────────────@───Rx(-0.5π)──I──────X0────────────────────
                                    │                              │              │      │
    1: ───X──────────────H──────────X───@──────────────────────@───X───H──────────I──────Y1────────────────────
                                        │                      │                  │      │
    2: ───H─────────────────────────────X───@──────────────@───X───H──────────────I──────Y2────────────────────
                                            │              │                      │      │
    3: ───H─────────────────────────────────X───Rz(2.0π)───X───H──────────────────1*I3───1j*Y3───────────────M─
                                                                                  │      │                   │
    4: ─── U = 1.17 rad ──────────────────────────────────────────────────────────(0)────@──── U = 1.17 rad ─M─

    gives:
            single and double =  ({1: 13, 2: 14}, 41)
                            aka 13 single qubit gates and 14 two qubit gates
            OR

            multi =  ({1: 13, 2: 6, 5: 2}, 35)
    """
    counter = {}

    if only_one_two_qubit_gates:
        ############################ decompose into only one and two qubit gates
        if M_gates_included:
            for op in list(Q_Circuit.all_operations())[:-1]:
                num_qubits = len(op.qubits)
                n_gates = 1

                if num_qubits > 2:
                    n_gates = num_qubits - 1
                    num_qubits = 2
                if num_qubits not in counter.keys():
                    counter[num_qubits] = n_gates
                else:
                    counter[num_qubits] += n_gates
        else:
            for op in Q_Circuit.all_operations():
                num_qubits = len(op.qubits)
                n_gates = 1

                if num_qubits > 2:
                    n_gates = num_qubits - 1
                    num_qubits = 2
                if num_qubits not in counter.keys():
                    counter[num_qubits] = n_gates
                else:
                    counter[num_qubits] += n_gates
    else:
        ########################### decompose into only any multi-qubit gate
        if M_gates_included:
            for op in list(Q_Circuit.all_operations())[:-1]:
                num_qubits = len(op.qubits)

                if num_qubits not in counter.keys():
                    counter[num_qubits] = 1
                else:
                    counter[num_qubits] += 1
        else:
            for op in Q_Circuit.all_operations():
                num_qubits = len(op.qubits)
                if num_qubits not in counter.keys():
                    counter[num_qubits] = 1
                else:
                    counter[num_qubits] += 1

    total_gates = sum(key * value for key, value in counter.items())

    return counter, total_gates

def Gate_Type_Count(Q_Circuit, M_gates_included=True):
    # note need __repr__ in custom gate classes to be defined!

    counter = {}
    if M_gates_included:
        for op in list(Q_Circuit.all_operations())[:-1]:
            gate = str(op.gate)

            if gate not in counter.keys():
                counter[gate] = 1
            else:
                counter[gate] += 1
    else:
        for op in Q_Circuit.all_operations():
            gate = str(op.gate)
            if gate not in counter.keys():
                counter[gate] = 1
            else:
                counter[gate] += 1
    return counter


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

    def __str__(self):
        return ' U = {} rad '.format(self.theta.__round__(4))

    def __repr__(self):
        return ' U_arb_state_prep'
def Get_control_parameters(num_qubits, Coefficient_list):
    if len(Coefficient_list) != 2 ** num_qubits:
        # fill missing terms with amplitude of zero!
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
                        # yield cirq.I.on(cirq.LineQubit(qubit+self.N_system_qubits))
                        pass
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

### breaking up multi-control gate into one and two qubit gates!
from functools import reduce
class My_V_gate(cirq.SingleQubitGate):
    """
    Description

    Args:
        theta (float): angle to rotate by in radians.
        number_control_qubits (int): number of control qubits
    """

    def __init__(self, V_mat, V_dag_mat, dagger_gate=False):
        self.V_mat = V_mat
        self.V_dag_mat = V_dag_mat
        self.dagger_gate = dagger_gate

    def _unitary_(self):
        if self.dagger_gate:
            return self.V_dag_mat
        else:
            return self.V_mat

    def num_qubits(self):
        return 1

    def _circuit_diagram_info_(self, args):
        if self.dagger_gate:
            return 'V^{†}'
        else:
            return 'V'

    def __str__(self):
        if self.dagger_gate:
            return 'V^{†}'
        else:
            return 'V'

    def __repr__(self):
        return self.__str__()
def int_to_Gray(base_10_num, n_qubits):
    # https://en.wikipedia.org/wiki/Gray_code

    # print(np.binary_repr(num, n_qubits)) # standard binary form!

    # The operator >> is shift right. The operator ^ is exclusive or
    gray_int = base_10_num ^ (base_10_num >> 1)

    return np.binary_repr(gray_int, n_qubits)
def check_binary_str_parity(binary_str):
    """
    Returns 0 for EVEN parity
    Returns 1 for ODD parity
    """
    parity = sum(map(int, binary_str)) % 2

    return parity


from sympy.physics.quantum import Dagger
import numpy as np
import cirq
from sympy import *
class n_control_U(cirq.Gate):
    """
    """

    def __init__(self, list_of_control_qubits, list_control_vals, U_qubit, U_cirq_gate, n_control_qubits):
        self.U_qubit = U_qubit
        self.U_cirq_gate = U_cirq_gate

        if len(list_of_control_qubits) != len(list_control_vals):
            raise ValueError('incorrect qubit control bits or incorrect number of control qubits')

        self.list_of_control_qubits = list_of_control_qubits
        self.list_control_vals = list_control_vals

        self.n_ancilla = len(list_of_control_qubits)
        self.D = None
        self.n_root = 2 ** (n_control_qubits - 1)
        self.n_control_qubits = n_control_qubits

        self.V_mat = None
        self.V_dag_mat = None

    def _diagonalise_U(self):

        # find diagonal matrix:
        U_matrix = Matrix(self.U_cirq_gate._unitary_())
        self.S, self.D = U_matrix.diagonalize()
        self.S_inv = self.S ** -1
        # where U = S D S^{-1}

        if not np.allclose(np.array(self.S * (self.D * self.S_inv), complex), self.U_cirq_gate._unitary_()):
            raise ValueError('U != SDS-1')

    def Get_V_gate_matrices(self, check=True):

        if self.D is None:
            self._diagonalise_U()
        D_nth_root = self.D ** (1 / self.n_root)

        V_mat = self.S * D_nth_root * self.S_inv
        V_dag_mat = Dagger(V_mat)

        self.V_mat = np.array(V_mat, complex)
        self.V_dag_mat = np.array(V_dag_mat, complex)

        if check:
            V_power_n = reduce(np.matmul, [self.V_mat for _ in range(self.n_root)])
            if not np.allclose(V_power_n, self.U_cirq_gate._unitary_()):
                raise ValueError('V^{n} != U')

    def flip_control_to_zero(self):
        for index, control_qubit in enumerate(self.list_of_control_qubits):
            if self.list_control_vals[index] == 0:
                yield cirq.X.on(control_qubit)

    def _get_gray_control_lists(self):

        grey_cntrl_bit_lists = []
        n_ancilla = len(self.list_of_control_qubits)
        for grey_index in range(1, 2 ** n_ancilla):
            gray_control_str = int_to_Gray(grey_index, n_ancilla)[::-1]  # note reversing order
            control_list = list(map(int, gray_control_str))
            parity = check_binary_str_parity(gray_control_str)

            grey_cntrl_bit_lists.append((control_list, parity))
        return grey_cntrl_bit_lists

    def _decompose_(self, qubits):
        if (self.V_mat is None) or (self.V_dag_mat is None):
            self.Get_V_gate_matrices()

        V_gate_DAGGER = My_V_gate(self.V_mat, self.V_dag_mat, dagger_gate=True)
        V_gate = My_V_gate(self.V_mat, self.V_dag_mat, dagger_gate=False)

        ## flip if controlled on zero
        X_flip = self.flip_control_to_zero()
        yield X_flip

        ## perform controlled gate
        n_ancilla = len(self.list_of_control_qubits)

        grey_control_lists = self._get_gray_control_lists()

        for control_index, binary_control_tuple in enumerate(grey_control_lists):

            binary_control_seq, parity = binary_control_tuple
            control_indices = np.where(np.array(binary_control_seq) == 1)[0]
            control_qubit = control_indices[-1]

            if parity == 1:
                gate = V_gate.controlled(num_controls=1, control_values=[1]).on(
                    self.list_of_control_qubits[control_qubit], self.U_qubit)
            else:
                gate = V_gate_DAGGER.controlled(num_controls=1, control_values=[1]).on(
                    self.list_of_control_qubits[control_qubit], self.U_qubit)

            if control_index == 0:
                yield gate
            else:
                for c_index in range(len(control_indices[:-1])):
                    yield cirq.CNOT(self.list_of_control_qubits[control_indices[c_index]],
                                    self.list_of_control_qubits[control_indices[c_index + 1]])
                yield gate
                for c_index in list(range(len(control_indices[:-1])))[::-1]:
                    yield cirq.CNOT(self.list_of_control_qubits[control_indices[c_index]],
                                    self.list_of_control_qubits[control_indices[c_index + 1]])

        ## unflip if controlled on zero
        X_flip = self.flip_control_to_zero()
        yield X_flip

    def _circuit_diagram_info_(self, args):
        #         return cirq.protocols.CircuitDiagramInfo(
        #             wire_symbols=tuple([*['@' if bit==1 else '(0)' for bit in self.list_control_vals],'U']),
        #             exponent=1)
        return cirq.protocols.CircuitDiagramInfo(
            wire_symbols=tuple(
                [*['@' if bit == 1 else '(0)' for bit in self.list_control_vals], self.U_cirq_gate.__str__()]),
            exponent=1)

    def num_qubits(self):
        return len(self.list_of_control_qubits) + 1  # (+1 for U_qubit)

    def check_Gate_gate_decomposition(self, tolerance=1e-9):
        """
        function compares single and two qubit gate construction of n-controlled-U
        against perfect n-controlled-U gate

        tolerance is how close unitary matrices are required

        """

        # decomposed into single and two qubit gates
        decomposed = self._decompose_(None)
        n_controlled_U_quantum_Circuit = cirq.Circuit(decomposed)

        #         print(n_controlled_U_quantum_Circuit)

        # perfect gate
        perfect_circuit_obj = self.U_cirq_gate.controlled(num_controls=self.n_control_qubits,
                                                          control_values=self.list_control_vals).on(
            *self.list_of_control_qubits, self.U_qubit)

        perfect_circuit = cirq.Circuit(perfect_circuit_obj)

        #         print(perfect_circuit)

        if not np.allclose(n_controlled_U_quantum_Circuit.unitary(), perfect_circuit.unitary(), atol=tolerance):
            raise ValueError('V^{n} != U')
        else:
            #             print('Correct decomposition')
            return True

class State_Prep_Circuit_one_two_qubit_gates(cirq.Gate):
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

    def __init__(self, circuit_param_dict,N_ancilla_qubits, N_system_qubits=0, check_gate_decomposition=True):

        self.circuit_param_dict = circuit_param_dict
        self.N_system_qubits = N_system_qubits
        self.check_gate_decomposition = check_gate_decomposition
        self.N_ancilla_qubits=N_ancilla_qubits

    def _decompose_(self, qubits):

        for qubit in self.circuit_param_dict:

            for term in self.circuit_param_dict[qubit]:
                if term['control_state']:
                    control_values = [int(bit) for bit in term['control_state']]
                    num_controls = len(control_values)
                    theta = term['angle']

                    if theta == 0:
                        # yield cirq.I.on(cirq.LineQubit(qubit+self.N_system_qubits))
                        pass
                    else:
                        U_single_qubit_gate = My_U_Gate(theta)
                        list_of_control_qubits = cirq.LineQubit.range(self.N_system_qubits,
                                                                      self.N_system_qubits + num_controls)
                        U_qubit = cirq.LineQubit(qubit+self.N_system_qubits)
                        Control_U_gate = n_control_U(list_of_control_qubits, control_values, U_qubit, U_single_qubit_gate, num_controls)

                        if self.check_gate_decomposition:
                            Control_U_gate.check_Gate_gate_decomposition()

                        yield cirq.Circuit(cirq.decompose_once(
                            (Control_U_gate(*cirq.LineQubit.range(Control_U_gate.num_qubits())))))
                        ## better print BUT mistake with .unitary() method
                        # yield cirq.Circuit((Control_U_gate(*list_of_control_qubits,U_qubit))) #

                else:
                    theta = term['angle']
                    if theta == 0:
                        # yield cirq.I.on(cirq.LineQubit(qubit+self.N_system_qubits))
                        pass
                    else:
                        yield My_U_Gate(theta).on(cirq.LineQubit(qubit+self.N_system_qubits))


    def _circuit_diagram_info_(self, args):

        max_qubit = max(self.circuit_param_dict.keys())
        string_list = []
        for _ in range(self.N_system_qubits):
            string_list.append('Do Nothing (system qubit)')
        for _ in range(self.N_ancilla_qubits):
            string_list.append('arb state prep')

        return string_list

    def num_qubits(self):
        return self.N_ancilla_qubits + self.N_system_qubits
## testing
# ancilla_amps=[np.sqrt(0.3), np.sqrt(0.1),np.sqrt(0.1),np.sqrt(0.1),np.sqrt(0.1),np.sqrt(0.1),np.sqrt(0.1),np.sqrt(0.1)]
# N_ancilla_qubits=int(np.ceil(np.log2(len(ancilla_amps))))
# N_system_qubits=3
#
# test1 = Get_control_parameters(3, ancilla_amps)
# test_circ = State_Prep_Circuit_one_two_qubit_gates(test1,N_system_qubits, N_system_qubits=N_system_qubits, check_gate_decomposition=True)
# print(cirq.Circuit((test_circ(*cirq.LineQubit.range(test_circ.num_qubits())))))
# print(cirq.Circuit(cirq.decompose_once(
#         (test_circ(*cirq.LineQubit.range(test_circ.num_qubits()))))))
# max([len(i) for i in list(test1.values())])//2

class prepare_arb_state_one_two_qubit_gates():
    def __init__(self, Coefficient_list, N_System_qubits):
        self.Coefficient_list = Coefficient_list
        self.N_System_qubits = N_System_qubits
    def _Get_control_parameters_dict(self):
        return Get_control_parameters(self.Get_max_no_ancilla_qubits(), self.Coefficient_list)

    def Get_state_prep_Circuit(self):
        circ_obj = State_Prep_Circuit_one_two_qubit_gates(self._Get_control_parameters_dict(), \
                                                          self.Get_max_no_ancilla_qubits(),N_system_qubits=self.N_System_qubits,check_gate_decomposition=True)
        circuit = cirq.Circuit(cirq.decompose_once((circ_obj(*cirq.LineQubit.range(circ_obj.num_qubits())))))
        return circuit

    def Get_max_no_ancilla_qubits(self):
        return int(np.ceil(np.log2(len(self.Coefficient_list))))  # note round up with np.ceil
# # testing
# ancilla_amps=[np.sqrt(0.3), np.sqrt(0.1),np.sqrt(0.1),np.sqrt(0.1),np.sqrt(0.1),np.sqrt(0.1),np.sqrt(0.1),np.sqrt(0.1)]
# N_ancilla_qubits=int(np.ceil(np.log2(len(ancilla_amps))))
# N_system_qubits=3
#
# vv = prepare_arb_state_one_two_qubit_gates(ancilla_amps, N_system_qubits)
# x= vv.Get_state_prep_Circuit()
# x.unitary()