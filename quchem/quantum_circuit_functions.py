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

        self.state = initial_state


    def _decompose_(self, qubits):
        """
        Args:
            qubits (cirq.line.line_qubit.LineQubit): cirq qubits (given to decompose method by cirq)

        Raises:
            ValueError: State not in occupation number basis

        Yields:
            cirq.circuits.circuit.Circuit: cirq circuit generator!
        """

        for i in range(len(self.state)):
            state = self.state[i]
            qubitNo = i

            if state == 1:
                yield cirq.X(qubits[qubitNo])

            if state != 0 and state !=1:
                raise ValueError('initial state not in correct format... qubit {} has value {} ' \
                                 '[instead of 0 or 1]'.format(i, state))

    def num_qubits(self):
        return len(self.state)

    def _circuit_diagram_info_(self, args):
        state_prep_list = []
        for i in range(len(self.state)):
            state = self.state[i]
            if state == 1:
                state_prep_list.append('state_prep: |1> ')
            elif state == 0:
                state_prep_list.append('state_prep: |0> ')
            else:
                raise ValueError('state needs to be list of 0 or 1 s ' \
                                 'qubit {} has value {}'.format(i, state))
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
                yield cirq.CNOT.on(qubits[qubitNo], qubits[qubitNo_next])

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

    e.g.: ('X0 I1 Y2 X3', 0.125j)
        gives :
                0: ───H───────────
                2: ───Rx(-0.5π)───
                3: ───H───────────

    Args:
        PauliWord_and_cofactor (tuple): Tuple of PauliWord (str) and constant (complex) ... (PauliWord, constant)

    Returns
        A cirq circuit object to be used by cirq.Circuit

    """
    def __init__(self, PauliWord_and_cofactor):

        self.PauliWord_and_cofactor = PauliWord_and_cofactor

    def _decompose_(self, qubits):

        PauliWord = self.PauliWord_and_cofactor[0].split(' ')

        for PauliString in PauliWord:
            qubitOp = PauliString[0]
            qubitNo = int(PauliString[1::])

            if qubitOp == 'X':
                yield cirq.H(qubits[qubitNo])
            elif qubitOp == 'Y':
                 yield cirq.rx(-np.pi / 2)(qubits[qubitNo])
            elif qubitOp == 'Z' or 'I':
                continue
            else:
                raise ValueError("Qubit Operation: {} is NOT a Pauli operation".format(qubitOp))

    def _circuit_diagram_info_(self, args):
        Ansatz_basis_change_list = []
        PauliWord = self.PauliWord_and_cofactor[0].split(' ')
        for i in range(len(PauliWord)):
                Ansatz_basis_change_list.append('Basis_change')
        return Ansatz_basis_change_list

    def num_qubits(self):
        PauliWord = self.PauliWord_and_cofactor[0].split(' ')
        return len(PauliWord)

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

def Generate_Full_Q_Circuit_of_Molecular_Hamiltonian(Full_Ansatz_Q_Circuit, PauliWord_str_list_Qubit_Hamiltonian, n_qubits):
    """
     Function that appends Ansatz Quantum Circuit to Pauli perform and measure circuit instance.

    Args:
        Full_Ansatz_Q_Circuit (cirq.circuits.circuit.Circuit): Full cirq Ansatz Q circuit
        PauliWord_str_list (list): List of tuples containing PauliWord to perform and constant -> (PauliWord, constant)
        n_qubits (int): number of qubits

    Returns:
        dic_holder (dict): Returns a dictionary of each quantum circuit, with cofactor, PauliWord and cirq Q Circuit

    """
    # dic_holder = []
    # for PauliString_and_Constant in PauliWord_str_list_Qubit_Hamiltonian:
    #     Q_circuit = Generate_Full_Q_Circuit(Full_Ansatz_Q_Circuit, PauliString_and_Constant)
    #     dic_holder.append([Q_circuit, PauliString_and_Constant[1]])

    I_Measure = ['I{}'.format(i) for i in range(n_qubits)]
    seperator = ' '
    PauliWord_I_only = seperator.join(I_Measure)

    dic_holder = {}
    for i in range(len(PauliWord_str_list_Qubit_Hamiltonian)):
        PauliString_and_Constant = PauliWord_str_list_Qubit_Hamiltonian[i]
        temp_d={}
        if PauliString_and_Constant[0] == PauliWord_I_only:
            temp_d['circuit'] = None
            temp_d['gamma_l'] = PauliString_and_Constant[1]
            temp_d['PauliWord'] = PauliString_and_Constant[0]
        else:
            Q_circuit = Generate_Full_Q_Circuit(Full_Ansatz_Q_Circuit, PauliString_and_Constant)
            temp_d['circuit'] = Q_circuit
            temp_d['gamma_l'] = PauliString_and_Constant[1]
            temp_d['PauliWord'] = PauliString_and_Constant[0]
        dic_holder[i] = temp_d

    return dic_holder



