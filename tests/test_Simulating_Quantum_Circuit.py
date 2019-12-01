from quchem.Simulating_Quantum_Circuit import *
from quchem.quantum_circuit_functions import *
import cirq
import pytest
# in terminal type: py.test -v

###
def test_Get_Histogram_key():
    """
    Standard use test
    """
    PauliWord = 'I0 Z1 Z2 I3 I4 X5'
    Histogram_key = Get_Histogram_key(PauliWord)
    expected = '1,2,5'

    assert expected and Histogram_key

###
def test_Simulate_Quantum_Circuit_PauliWord():
    """

    """
    num_shots = 10000
    PauliWord_and_cofactor = ('Z0 X1', -0.28527408634774526j)
    expected = {1: num_shots/2, 0: num_shots/2}

    circuit_gen = Perform_PauliWord_and_Measure(PauliWord_and_cofactor)

    quantum_circuit = cirq.Circuit.from_ops(cirq.decompose_once((circuit_gen(*cirq.LineQubit.range(circuit_gen.num_qubits())))))

    histogram_string = Get_Histogram_key(PauliWord_and_cofactor[0])

    counter = dict(Simulate_Quantum_Circuit(quantum_circuit, num_shots, histogram_string))

    check_list=[]
    for key in counter:
        check_list.append(np.isclose(counter[key], expected[key], rtol=100))

    assert all(check_list)

def test_Simulate_Quantum_Circuit_single_H():
    """

    """
    num_shots = 10000
    expected = {1: num_shots/2, 0: num_shots/2}

    qubit = cirq.LineQubit.range(1)
    quantum_circuit = cirq.Circuit.from_ops([cirq.H.on(*qubit),
                           cirq.measure(*qubit)])

    histogram_string = '0' #only measurement on line qubit 0

    counter = dict(Simulate_Quantum_Circuit(quantum_circuit, num_shots, histogram_string))

    check_list=[]
    for key in counter:
        check_list.append(np.isclose(counter[key], expected[key], rtol=100))

    assert all(check_list)
###

def test_Return_as_binary_PauliWord():
    """

    :return:
    """
    PauliWord_and_cofactor = ('Z0 X1 Y2 I3 X4', -0.28527408634774526j)
    num_shots = 1000

    circuit_gen = Perform_PauliWord_and_Measure(PauliWord_and_cofactor)
    quantum_circuit = cirq.Circuit.from_ops(cirq.decompose_once((circuit_gen(*cirq.LineQubit.range(circuit_gen.num_qubits())))))
    histogram_string = Get_Histogram_key(PauliWord_and_cofactor[0])

    counter_result = Simulate_Quantum_Circuit(quantum_circuit, num_shots, histogram_string)
    binary_counter_result = Return_as_binary(counter_result, PauliWord_and_cofactor[0])

    non_I = [i for i in PauliWord_and_cofactor[0].split(' ') if i[0] != 'I']
    binary_length = '{' + '0:0{}b'.format(len(non_I)) + '}'
    expected={}
    for key in counter_result:
        new_key = binary_length.format(key)
        expected[new_key] = counter_result[key]

    assert expected == binary_counter_result

def test_Return_as_binary_double_H():
    """

    :return:
    """

    qubits = cirq.LineQubit.range(2)
    quantum_circuit = cirq.Circuit.from_ops([cirq.H.on(qubits[0]), cirq.H.on(qubits[1]),
                           cirq.measure(*qubits)])

    histogram_string = '0,1' # measurement on line qubits 0 and 1

    num_shots = 1000
    counter_result = Simulate_Quantum_Circuit(quantum_circuit, num_shots, histogram_string)

    Gates = 'H0, H1'
    binary_counter_result = Return_as_binary(counter_result, Gates)

    binary_length = '{' + '0:0{}b'.format(len(Gates.split(' '))) + '}'
    expected={}
    for key in counter_result:
        new_key = binary_length.format(key)
        expected[new_key] = counter_result[key]

    assert expected == binary_counter_result

###


def test_calc_parity_PauliWord():
    PauliWord_and_cofactor = ('Z0 X1 Y2 I3 X4', -0.28527408634774526j)
    num_shots = 1000
    circuit_gen = Perform_PauliWord_and_Measure(PauliWord_and_cofactor)
    quantum_circuit = cirq.Circuit.from_ops(cirq.decompose_once((circuit_gen(*cirq.LineQubit.range(circuit_gen.num_qubits())))))
    histogram_string = Get_Histogram_key(PauliWord_and_cofactor[0])
    counter_result = Simulate_Quantum_Circuit(quantum_circuit, num_shots, histogram_string)
    binary_counter_result = Return_as_binary(counter_result, PauliWord_and_cofactor[0])

    test={}
    expected = {}
    for state in binary_counter_result:
        test[state] = calc_parity(state)

        sum_of_bits = sum([int(bit) for bit in state])
        parity = sum_of_bits%2
        expected[state] = parity

    assert expected == test

def test_calc_parity_incorrect_quantum_State():
    quantum_state = '5500'
    with pytest.raises(ValueError) as exc_info:
        assert exc_info is calc_parity(quantum_state)

###
def test_expectation_value_by_parity_PauliWord():
    PauliWord_and_cofactor = ('Z0 X1 Y2 I3 X4', -0.28527408634774526j)
    num_shots = 1000
    circuit_gen = Perform_PauliWord_and_Measure(PauliWord_and_cofactor)
    quantum_circuit = cirq.Circuit.from_ops(cirq.decompose_once((circuit_gen(*cirq.LineQubit.range(circuit_gen.num_qubits())))))
    histogram_string = Get_Histogram_key(PauliWord_and_cofactor[0])
    counter_result = Simulate_Quantum_Circuit(quantum_circuit, num_shots, histogram_string)
    binary_counter_result = Return_as_binary(counter_result, PauliWord_and_cofactor[0])

    expectation_value = expectation_value_by_parity(binary_counter_result)


    Total = 0
    for state in binary_counter_result:
        parity = calc_parity(state)
        if parity == 0:
            Total += binary_counter_result[state]
        elif parity == 1:
            Total -= binary_counter_result[state]
    expected = Total / num_shots

    assert expectation_value == expected

def test_expectation_value_by_parity_double_H():

    qubits = cirq.LineQubit.range(2)
    quantum_circuit = cirq.Circuit.from_ops([cirq.H.on(qubits[0]), cirq.H.on(qubits[1]), cirq.measure(*qubits)])
    histogram_string = '0,1' # measurement on line qubits 0 and 1
    num_shots = 10000
    counter_result = Simulate_Quantum_Circuit(quantum_circuit, num_shots, histogram_string)
    Gates = 'H0, H1'
    binary_counter_result = Return_as_binary(counter_result, Gates)

    expectation_value = expectation_value_by_parity(binary_counter_result)

    Total = 0
    for state in binary_counter_result:
        parity = calc_parity(state)
        if parity == 0:
            Total += binary_counter_result[state]
        elif parity == 1:
            Total -= binary_counter_result[state]
    expected = Total / num_shots

    assert expectation_value == expected