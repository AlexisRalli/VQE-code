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
