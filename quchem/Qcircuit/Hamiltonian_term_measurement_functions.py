import cirq
import numpy as np


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