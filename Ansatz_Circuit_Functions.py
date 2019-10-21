import cirq
from Ansatz_Generator_Functions import *

class State_Prep(cirq.Gate):

    def __init__(self, initial_state):
        """""
    Circuit to obtain given initial state
    e.g. |0011>  =  [0,0,1,1]

    :param initial_state: A list description of HF state... note that indexing from far right to left.
    :type initial_state: list, (numpy.array, tuple)

    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: A circuit object to be used by cirq.Circuit.from_ops
    :rtype: class

        """
        self.state = initial_state


    def _decompose_(self, qubits):

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
                state_prep_list.append('state_prep: |1>')
            elif state == 0:
                state_prep_list.append('state_prep: |0>')
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

# Test
if __name__ == '__main__':
    initial_state = State_Prep([0,0,1,1])
    print(cirq.Circuit.from_ops((initial_state(*cirq.LineQubit.range(initial_state.num_qubits())))))
    print(
        cirq.Circuit.from_ops(cirq.decompose_once((initial_state(*cirq.LineQubit.range(initial_state.num_qubits()))))))



class Ansatz_basis_change_initial(cirq.Gate):
    def __init__(self, Qubit_NoOp_const_term):

        self.Qubit_NoOp_const_term = Qubit_NoOp_const_term
        self.num_qubits = len(Qubit_NoOp_const_term)
        #self.Num_line_qubits =

    def _decompose_(self, qubits):

        for qubitNo, qubitOp in self.Qubit_NoOp_const_term[0]:
            if qubitOp == 'X':
                yield cirq.H(qubits[qubitNo])
            elif qubitOp == 'Y':
                yield cirq.Rx(math.pi / 2)(qubits[qubitNo])
            elif qubitOp == 'Z':
                continue
            else:
                raise ValueError("Qubit Operation {} on qubit {} isn't a Pauli operation".format(qubitOp, qubitNo))



    def _circuit_diagram_info_(self, args):

        Ansatz_basis_change_list = []
        for i in range(self.Num_line_qubits()):
                Ansatz_basis_change_list.append('Ansatz_basis_change')

        return Ansatz_basis_change_list

    def num_qubits(self):
        return self.qubitNo

    def Num_line_qubits(self):

        indexes = [qubitNo for qubitNo, qubitOp in self.Qubit_NoOp_const_term[0]]
        highest_index = max(indexes)

        return highest_index + 1



if __name__ == '__main__':
    HF_initial_state = [0, 0, 1, 1]
    UCC = UCC_Terms(HF_initial_state)

    test_term = UCC.T1_formatted[0][0]

    circuit = Ansatz_basis_change_initial(test_term)

    print(cirq.Circuit.from_ops((circuit(*cirq.LineQubit.range(circuit.Num_line_qubits())))))
    print(
        cirq.Circuit.from_ops(cirq.decompose_once((circuit(*cirq.LineQubit.range(circuit.Num_line_qubits()))))))
    print(test_term)
