import cirq
import numpy as np


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

# Test
if __name__ == '__main__':
    initial_state = State_Prep([0,0,1,1])
    print(cirq.Circuit.from_ops((initial_state(*cirq.LineQubit.range(initial_state.num_qubits())))))
    print(
        cirq.Circuit.from_ops(cirq.decompose_once((initial_state(*cirq.LineQubit.range(initial_state.num_qubits()))))))

##
class Change_of_Basis_initial(cirq.Gate):
    def __init__(self, PauliWord_and_cofactor):
        """
         Circuit to perform change of basis in order to perform: e^(-i theta_sk/2 X_sk)
         This cirq gate changes to correct basis.

        :param PauliWord_and_cofactor: (PauliWord, constant)
        :type PauliWord_and_cofactor: tuple

        e.g.: ('X0 X1 X2 Y3', -0.28527408634774526j)

        ...
        :raises [ErrorType]: [ErrorDescription]
        ...
        :return: A circuit object to be used by cirq.Circuit.from_ops
        :rtype: class
       """

        self.PauliWord_and_cofactor = PauliWord_and_cofactor

    def _decompose_(self, qubits):

        PauliWord = self.PauliWord_and_cofactor[0].split(' ')

        for PauliString in PauliWord:
            qubitOp = PauliString[0]
            qubitNo = int(PauliString[1::])

            if qubitOp == 'X':
                yield cirq.H(qubits[qubitNo])
            elif qubitOp == 'Y':
                 yield cirq.Rx(np.pi / 2)(qubits[qubitNo])
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

if __name__ == '__main__':
    PauliWord_test = ('Z0 X1 Y2 Z3 I4 I5 I6 I7 I8 Y9 X10', (0.8918294488900189+0j))

    Basis_change_circuit = Change_of_Basis_initial(PauliWord_test)

    print(cirq.Circuit.from_ops((Basis_change_circuit(*cirq.LineQubit.range(Basis_change_circuit.num_qubits())))))
    print(
        cirq.Circuit.from_ops(cirq.decompose_once((Basis_change_circuit(*cirq.LineQubit.range(Basis_change_circuit.num_qubits()))))))

class Engtangle_initial(cirq.Gate):
    def __init__(self, PauliWord_and_cofactor):
        """
         Circuit to perform change of basis in order to perform: e^(-i theta_sk/2 X_sk)
         This cirq gate entangles qubits prior to rotation

        :param PauliWord_and_cofactor: (PauliWord, constant)
        :type PauliWord_and_cofactor: tuple

        e.g.: ('X0 X1 X2 Y3', -0.28527408634774526j)

        ...
        :raises [ErrorType]: [ErrorDescription]
        ...
        :return: A circuit object to be used by cirq.Circuit.from_ops
        :rtype: class
       """
        self.PauliWord_and_cofactor = PauliWord_and_cofactor

    def _decompose_(self, qubits):

        PauliWord = self.PauliWord_and_cofactor[0].split(' ')

        # note identity terms removed here
        qubitNo_qubitOp_list = [(int(PauliString[1::]), PauliString[0]) for PauliString in PauliWord if PauliString[0] != 'I']

        control_qubit = max([qubitNo for qubitNo, qubitOp in qubitNo_qubitOp_list])

        for j in range(len(qubitNo_qubitOp_list)):
            qubitNo = qubitNo_qubitOp_list[j][0]
            #qubitOp = qubitNo_qubitOp_list[j][1]

            if qubitNo < control_qubit:
                qubitNo_NEXT = qubitNo_qubitOp_list[j + 1][0]
                yield cirq.CNOT(qubits[qubitNo], qubits[qubitNo_NEXT])


    def _circuit_diagram_info_(self, args):
        string_list = []
        PauliWord = self.PauliWord_and_cofactor[0].split(' ')
        for i in range(len(PauliWord)):
                string_list.append('Entangling circuit')
        return string_list


    def num_qubits(self):
        PauliWord = self.PauliWord_and_cofactor[0].split(' ')
        return len(PauliWord)

if __name__ == '__main__':
    PauliWord_test = ('Z0 X1 Y2 Z3 I4 I5 I6 I7 I8 Y9 X10', (0.8918294488900189+0j))
    Ent_initial = Engtangle_initial(PauliWord_test)

    print(cirq.Circuit.from_ops((Ent_initial(*cirq.LineQubit.range(Ent_initial.num_qubits())))))
    print(
        cirq.Circuit.from_ops(cirq.decompose_once((Ent_initial(*cirq.LineQubit.range(Ent_initial.num_qubits()))))))

class PauliWord_exponential_rotation(cirq.Gate):

    def __init__(self, PauliWord_and_cofactor, theta):
        """
         Circuit to perform change of basis in order to perform: e^(-i theta_sk/2 X_sk)
         This cirq gate entangles qubits prior to rotation

        :param PauliWord_and_cofactor: (PauliWord, constant)
        :type PauliWord_and_cofactor: tuple

         e.g.: ('X0 X1 X2 Y3', -0.28527408634774526j)

        :param theta: angle to rotate by
        :type theta: float

        ...
        :raises [ErrorType]: [ErrorDescription]
        ...
        :return: A circuit object to be used by cirq.Circuit.from_ops
        :rtype: class
       """
        self.PauliWord_and_cofactor = PauliWord_and_cofactor
        self.theta = theta


    def _decompose_(self, qubits):

        PauliWord = self.PauliWord_and_cofactor[0].split(' ')

        # note identity terms removed here
        qubitNo_qubitOp_list = [(int(PauliString[1::]), PauliString[0]) for PauliString in PauliWord if PauliString[0] != 'I']

        control_qubit = max([qubitNo for qubitNo, qubitOp in qubitNo_qubitOp_list])

        yield cirq.Rz(self.theta).on(qubits[control_qubit])

    def num_qubits(self):
        PauliWord = self.PauliWord_and_cofactor[0].split(' ')
        return len(PauliWord)

    def _circuit_diagram_info_(self, args):
        string_list = []
        PauliWord = self.PauliWord_and_cofactor[0].split(' ')
        for i in range(len(PauliWord)):
                string_list.append('R_sk_rotation circuit')
        return string_list

if __name__ == '__main__':
    PauliWord_test = ('Z0 X1 Y2 Z3 I4 I5 I6 I7 I8 Y9 X10', (0.8918294488900189+0j))
    theta = np.pi

    Rzz = PauliWord_exponential_rotation(PauliWord_test, theta)

    print(cirq.Circuit.from_ops((Rzz(*cirq.LineQubit.range(Rzz.num_qubits())))))
    print(
        cirq.Circuit.from_ops(cirq.decompose_once((Rzz(*cirq.LineQubit.range(Rzz.num_qubits()))))))

class Engtangle_final(cirq.Gate):
    def __init__(self, PauliWord_and_cofactor):
        """
         Circuit to perform change of basis in order to perform: e^(-i theta_sk/2 X_sk)
         This cirq gate entangles qubits post rotation

        :param PauliWord_and_cofactor: (PauliWord, constant)
        :type PauliWord_and_cofactor: tuple

        e.g.: ('X0 X1 X2 Y3', -0.28527408634774526j)

        ...
        :raises [ErrorType]: [ErrorDescription]
        ...
        :return: A circuit object to be used by cirq.Circuit.from_ops
        :rtype: class
       """
        self.PauliWord_and_cofactor = PauliWord_and_cofactor

    def _decompose_(self, qubits):

        PauliWord = self.PauliWord_and_cofactor[0].split(' ')

        # note identity terms removed here
        qubitNo_qubitOp_list_REVERSE = [(int(PauliString[1::]), PauliString[0]) for PauliString in PauliWord if PauliString[0] != 'I'][::-1]

        control_qubit = max([qubitNo for qubitNo, qubitOp in qubitNo_qubitOp_list_REVERSE])

        for i in range(len(qubitNo_qubitOp_list_REVERSE)):
            qubitNo, qubitOp = qubitNo_qubitOp_list_REVERSE[i]

            if qubitNo < control_qubit and qubitNo >= 0:
                qubitNo_NEXT = qubitNo_qubitOp_list_REVERSE[i - 1][0]   # note negative here
                yield cirq.CNOT(qubits[qubitNo], qubits[qubitNo_NEXT])


    def _circuit_diagram_info_(self, args):
        string_list = []
        PauliWord = self.PauliWord_and_cofactor[0].split(' ')
        for i in range(len(PauliWord)):
                string_list.append('Entangling circuit')
        return string_list


    def num_qubits(self):
        PauliWord = self.PauliWord_and_cofactor[0].split(' ')
        return len(PauliWord)

if __name__ == '__main__':
    PauliWord_test = ('Z0 X1 Y2 Z3 I4 I5 I6 I7 I8 Y9 X10', (0.8918294488900189+0j))
    Ent_final = Engtangle_final(PauliWord_test)

    print(cirq.Circuit.from_ops((Ent_final(*cirq.LineQubit.range(Ent_final.num_qubits())))))
    print(
        cirq.Circuit.from_ops(cirq.decompose_once((Ent_final(*cirq.LineQubit.range(Ent_final.num_qubits()))))))

class Change_of_Basis_final(cirq.Gate):
    def __init__(self, PauliWord_and_cofactor):
        """
         Circuit to perform change of basis in order to perform: e^(-i theta_sk/2 X_sk)
         This cirq gate changes to correct basis.

        :param PauliWord_and_cofactor: (PauliWord, constant)
        :type PauliWord_and_cofactor: tuple

        e.g.: ('X0 X1 X2 Y3', -0.28527408634774526j)

        ...
        :raises [ErrorType]: [ErrorDescription]
        ...
        :return: A circuit object to be used by cirq.Circuit.from_ops
        :rtype: class
       """

        self.PauliWord_and_cofactor = PauliWord_and_cofactor

    def _decompose_(self, qubits):

        PauliWord = self.PauliWord_and_cofactor[0].split(' ')

        for PauliString in PauliWord:
            qubitOp = PauliString[0]
            qubitNo = int(PauliString[1::])

            if qubitOp == 'X':
                yield cirq.H(qubits[qubitNo])
            elif qubitOp == 'Y':
                 yield cirq.Rx(-np.pi / 2)(qubits[qubitNo])
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

if __name__ == '__main__':
    PauliWord_test = ('Z0 X1 Y2 Z3 I4 I5 I6 I7 I8 Y9 X10', (0.8918294488900189+0j))

    Basis_change_circuit = Change_of_Basis_final(PauliWord_test)

    print(cirq.Circuit.from_ops((Basis_change_circuit(*cirq.LineQubit.range(Basis_change_circuit.num_qubits())))))
    print(
        cirq.Circuit.from_ops(cirq.decompose_once((Basis_change_circuit(*cirq.LineQubit.range(Basis_change_circuit.num_qubits()))))))

class full_exponentiated_PauliWord_circuit(cirq.Gate):
    def __init__(self, PauliWord_and_cofactor, theta):
        """
        This circuit takes in PauliWord and angle and performs: e^(-i theta_sk/2 X_sk)

        :param PauliWord_and_cofactor: (PauliWord, constant)
        :type PauliWord_and_cofactor: tuple

        :param theta: angle to rotate by
        :type theta: float

        e.g.: ('X0 X1 X2 Y3', -0.28527408634774526j)

        ...
        :raises [ErrorType]: [ErrorDescription]
        ...
        :return: A circuit object to be used by cirq.Circuit.from_ops
        :rtype: class
       """
        self.PauliWord_and_cofactor = PauliWord_and_cofactor
        self.theta = theta


    def _decompose_(self, qubits):


        Basis_change_initial_circuit = Change_of_Basis_initial(self.PauliWord_and_cofactor)
        Ent_initial = Engtangle_initial(self.PauliWord_and_cofactor)
        PauliWord_exponential_rotation_circuit = PauliWord_exponential_rotation(self.PauliWord_and_cofactor, self.theta)
        Ent_final = Engtangle_final(self.PauliWord_and_cofactor)
        Basis_change_final_circuit = Change_of_Basis_final(self.PauliWord_and_cofactor)

        basis_change_initial_gen = Basis_change_initial_circuit._decompose_(qubits)
        Ent_initial_gen = Ent_initial._decompose_(qubits)
        PauliWord_exponential_rotation_circuit_gen = PauliWord_exponential_rotation_circuit._decompose_(qubits)
        Ent_final_gen = Ent_final._decompose_(qubits)
        basis_change_final_gen = Basis_change_final_circuit._decompose_(qubits)

        list_generators = [basis_change_initial_gen, Ent_initial_gen, PauliWord_exponential_rotation_circuit_gen, Ent_final_gen,
                           basis_change_final_gen]
        yield list_generators



    def _circuit_diagram_info_(self, args):
        string_list = []
        PauliWord = self.PauliWord_and_cofactor[0].split(' ')
        for i in range(len(PauliWord)):
                string_list.append('full_exp_PauliWord_circuit')
        return string_list

    def num_qubits(self):
        PauliWord = self.PauliWord_and_cofactor[0].split(' ')
        return len(PauliWord)

if __name__ == '__main__':
    PauliWord_test = ('Z0 X1 Y2 Z3 I4 I5 I6 I7 I8 Y9 X10', (0.8918294488900189+0j))
    theta = np.pi

    full_Oper = full_exponentiated_PauliWord_circuit(PauliWord_test, theta)

    print(cirq.Circuit.from_ops((full_Oper(*cirq.LineQubit.range(full_Oper.num_qubits())))))
    print(
        cirq.Circuit.from_ops(
            cirq.decompose_once((full_Oper(*cirq.LineQubit.range(full_Oper.num_qubits()))))))


class Perform_PauliWord(cirq.Gate):
    def __init__(self, PauliWord_and_cofactor):
        """
        NOTE this is NOT the exponentiated operator: e^(-i theta/2 PAULI)

        This class builds a quantum circuit that performs a PauliWord

        :param PauliWord_and_cofactor: (PauliWord, constant)
        :type PauliWord_and_cofactor: tuple

        e.g.: ('X0 X1 X2 Y3', -0.28527408634774526j)

        ...
        :raises [ErrorType]: [ErrorDescription]
        ...
        :return: A circuit object to be used by cirq.Circuit.from_ops
        :rtype: class
       """

        self.PauliWord_and_cofactor = PauliWord_and_cofactor

    def _decompose_(self, qubits):

        PauliWord = self.PauliWord_and_cofactor[0].split(' ')

        for PauliString in PauliWord:
            qubitOp = PauliString[0]
            qubitNo = int(PauliString[1::])

            if qubitOp == 'X':
                yield cirq.X(qubits[qubitNo])
            elif qubitOp == 'Y':
                 yield cirq.Y(qubits[qubitNo])
            elif qubitOp == 'Z':
                yield cirq.Z(qubits[qubitNo])
            elif qubitOp == 'I':
                continue
            else:
                raise ValueError("Qubit Operation: {} is NOT a Pauli operation".format(qubitOp))

    def _circuit_diagram_info_(self, args):
        string_list = []
        PauliWord = self.PauliWord_and_cofactor[0].split(' ')
        for i in range(len(PauliWord)):
                string_list.append(' Performing_PauliWord ')
        return string_list

    def num_qubits(self):
        PauliWord = self.PauliWord_and_cofactor[0].split(' ')
        return len(PauliWord)

if __name__ == '__main__':
    PauliWord_test = ('Z0 X1 Y2 Z3 I4 I5 I6 I7 I8 Y9 X10', (0.8918294488900189+0j))
    Pauilword_circuit = Perform_PauliWord(PauliWord_test)

    print(cirq.Circuit.from_ops((Pauilword_circuit(*cirq.LineQubit.range(Pauilword_circuit.num_qubits())))))
    print(
        cirq.Circuit.from_ops(
            cirq.decompose_once((Pauilword_circuit(*cirq.LineQubit.range(Pauilword_circuit.num_qubits()))))))


class Change_Basis_to_Measure_PauliWord(cirq.Gate):
    def __init__(self, PauliWord_and_cofactor):
        """
        NOTE this is NOT the exponentiated operator: e^(-i theta/2 PAULI)

        This class builds a quantum circuit that performs a PauliWord

        :param PauliWord_and_cofactor: (PauliWord, constant)
        :type PauliWord_and_cofactor: tuple

        e.g.: ('X0 X1 X2 Y3', -0.28527408634774526j)

        ...
        :raises [ErrorType]: [ErrorDescription]
        ...
        :return: A circuit object to be used by cirq.Circuit.from_ops
        :rtype: class
       """

        self.PauliWord_and_cofactor = PauliWord_and_cofactor

    def _decompose_(self, qubits):

        PauliWord = self.PauliWord_and_cofactor[0].split(' ')

        for PauliString in PauliWord:
            qubitOp = PauliString[0]
            qubitNo = int(PauliString[1::])

            if qubitOp == 'X':
                yield cirq.H(qubits[qubitNo])
            elif qubitOp == 'Y':
                yield (cirq.S**-1)(qubits[qubitNo])
            elif qubitOp == 'Z':
                continue
            elif qubitOp == 'I':
                continue
            else:
                raise ValueError("Qubit Operation: {} is NOT a Pauli operation".format(qubitOp))

    def _circuit_diagram_info_(self, args):
        string_list = []
        PauliWord = self.PauliWord_and_cofactor[0].split(' ')
        for i in range(len(PauliWord)):
            string_list.append(' Changing_Basis_to_Measure_PauliWord ')
        return string_list

    def num_qubits(self):
        PauliWord = self.PauliWord_and_cofactor[0].split(' ')
        return len(PauliWord)

class Measure_PauliWord(cirq.Gate):
    def __init__(self, PauliWord_and_cofactor):
        """
        NOTE this is NOT the exponentiated operator: e^(-i theta/2 PAULI)

        This class builds a quantum circuit that performs a PauliWord

        :param PauliWord_and_cofactor: (PauliWord, constant)
        :type PauliWord_and_cofactor: tuple

        e.g.: ('X0 X1 X2 Y3', -0.28527408634774526j)

        ...
        :raises [ErrorType]: [ErrorDescription]
        ...
        :return: A circuit object to be used by cirq.Circuit.from_ops
        :rtype: class
       """

        self.PauliWord_and_cofactor = PauliWord_and_cofactor

    def _decompose_(self, qubits):

        PauliWord = self.PauliWord_and_cofactor[0].split(' ')

        qubits_to_measure = []  # list of line qubits to measure!
        list_of_qubitNo_to_measure = []
        for PauliString in PauliWord:
            qubitOp = PauliString[0]
            qubitNo = int(PauliString[1::])

            if qubitOp in ['X', 'Y', 'Z']:
                qubits_to_measure.append(qubits[qubitNo])
                list_of_qubitNo_to_measure.append(str(qubitNo))
            elif qubitOp == 'I':
                continue
            else:
                raise ValueError("Qubit Operation: {} is NOT a Pauli operation".format(qubitOp))

        string = ','.join(list_of_qubitNo_to_measure)

        # note to use cirq.measurementgate...
        # need string = '0,1,2,3'
        # qubits to measure =  [cirq.LineQubit(0), cirq.LineQubit(1), cirq.LineQubit(2), cirq.LineQubit(3)]

        if qubits_to_measure != []:
            yield cirq.MeasurementGate(string, ()).on(*qubits_to_measure)
        else:
            return None



    def _circuit_diagram_info_(self, args):
        string_list = []
        PauliWord = self.PauliWord_and_cofactor[0].split(' ')
        for i in range(len(PauliWord)):
            string_list.append(' Measuring_PauliWord ')
        return string_list

    def num_qubits(self):
        PauliWord = self.PauliWord_and_cofactor[0].split(' ')
        return len(PauliWord)

if __name__ == '__main__':
    PauliWord_test = ('Z0 X1 Y2 Z3 I4 I5 I6 I7 I8 Y9 X10', (0.8918294488900189+0j))
    Pauilword_Measure_circuit = Measure_PauliWord(PauliWord_test)

    #print(cirq.Circuit.from_ops((Pauilword_Measure_circuit(*cirq.LineQubit.range(Pauilword_Measure_circuit.num_qubits())))))
    print(
        cirq.Circuit.from_ops(
            cirq.decompose_once((Pauilword_Measure_circuit(*cirq.LineQubit.range(Pauilword_Measure_circuit.num_qubits()))))))


class Change_Basis_and_Measure_PauliWord(cirq.Gate):
    def __init__(self, PauliWord_and_cofactor):
        """
        blah

        :param PauliWord_and_cofactor: (PauliWord, constant)
        :type PauliWord_and_cofactor: tuple

        :param theta: angle to rotate by
        :type theta: float

        e.g.: ('X0 X1 X2 Y3', -0.28527408634774526j)

        ...
        :raises [ErrorType]: [ErrorDescription]
        ...
        :return: A circuit object to be used by cirq.Circuit.from_ops
        :rtype: class
       """
        self.PauliWord_and_cofactor = PauliWord_and_cofactor


    def _decompose_(self, qubits):


        change_basis_to_measure_circuit = Change_Basis_to_Measure_PauliWord(self.PauliWord_and_cofactor)
        measurment_circuit = Measure_PauliWord(self.PauliWord_and_cofactor)


        change_basis_to_measure_circuit_gen = change_basis_to_measure_circuit._decompose_(qubits)
        measurment_circuit_gen = measurment_circuit._decompose_(qubits)


        list_generators = [change_basis_to_measure_circuit_gen, measurment_circuit_gen]
        yield list_generators


    def _circuit_diagram_info_(self, args):
        string_list = []
        PauliWord = self.PauliWord_and_cofactor[0].split(' ')
        for i in range(len(PauliWord)):
                string_list.append('change_Basis_PauliWord_and_measure')
        return string_list

    def num_qubits(self):
        PauliWord = self.PauliWord_and_cofactor[0].split(' ')
        return len(PauliWord)

if __name__ == '__main__':
    PauliWord_test = ('Z0 X1 Y2 Z3 I4 I5 I6 I7 I8 Y9 X10', (0.8918294488900189+0j))
    Pauilword_FULL_circuit = Change_Basis_and_Measure_PauliWord(PauliWord_test)

    print(cirq.Circuit.from_ops((Pauilword_FULL_circuit(*cirq.LineQubit.range(Pauilword_FULL_circuit.num_qubits())))))
    print(
        cirq.Circuit.from_ops(
            cirq.decompose_once((Pauilword_FULL_circuit(*cirq.LineQubit.range(Pauilword_FULL_circuit.num_qubits()))))))


class Perform_PauliWord_and_Measure(cirq.Gate):
    def __init__(self, PauliWord_and_cofactor):
        """
        blah

        :param PauliWord_and_cofactor: (PauliWord, constant)
        :type PauliWord_and_cofactor: tuple

        :param theta: angle to rotate by
        :type theta: float

        e.g.: ('X0 X1 X2 Y3', -0.28527408634774526j)

        ...
        :raises [ErrorType]: [ErrorDescription]
        ...
        :return: A circuit object to be used by cirq.Circuit.from_ops
        :rtype: class
       """
        self.PauliWord_and_cofactor = PauliWord_and_cofactor


    def _decompose_(self, qubits):


        Perform_PauliWord_circuit = Perform_PauliWord(self.PauliWord_and_cofactor)
        change_basis_to_measure_circuit = Change_Basis_to_Measure_PauliWord(self.PauliWord_and_cofactor)
        measurment_circuit = Measure_PauliWord(self.PauliWord_and_cofactor)


        Perform_PauliWord_circuit_gen = Perform_PauliWord_circuit._decompose_(qubits)
        change_basis_to_measure_circuit_gen = change_basis_to_measure_circuit._decompose_(qubits)
        measurment_circuit_gen = measurment_circuit._decompose_(qubits)


        list_generators = [Perform_PauliWord_circuit_gen, change_basis_to_measure_circuit_gen, measurment_circuit_gen]
        yield list_generators


    def _circuit_diagram_info_(self, args):
        string_list = []
        PauliWord = self.PauliWord_and_cofactor[0].split(' ')
        for i in range(len(PauliWord)):
                string_list.append('Perform_PauliWord_and_measure')
        return string_list

    def num_qubits(self):
        PauliWord = self.PauliWord_and_cofactor[0].split(' ')
        return len(PauliWord)

if __name__ == '__main__':
    PauliWord_test = ('Z0 X1 Y2 Z3 I4 I5 I6 I7 I8 Y9 X10', (0.8918294488900189+0j))
    Pauilword_FULL_circuit = Perform_PauliWord_and_Measure(PauliWord_test)

    print(cirq.Circuit.from_ops((Pauilword_FULL_circuit(*cirq.LineQubit.range(Pauilword_FULL_circuit.num_qubits())))))
    print(
        cirq.Circuit.from_ops(
            cirq.decompose_once((Pauilword_FULL_circuit(*cirq.LineQubit.range(Pauilword_FULL_circuit.num_qubits()))))))
