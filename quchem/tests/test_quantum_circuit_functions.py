if __name__ == '__main__':
    from VQE_methods.quantum_circuit_functions import *
else:
    from .VQE_methods.quantum_circuit_functions import *

import cirq
import pytest
# in terminal type: py.test -v

def test_State_Prep():
    """
    Standard use test
    """

    working_circuit_type = type(cirq.Circuit.from_ops(cirq.H(1)))


    initial_state = State_Prep([0, 0, 1, 1])
    circuit1 = cirq.Circuit.from_ops((initial_state(*cirq.LineQubit.range(initial_state.num_qubits()))))
    circuit2 = cirq.Circuit.from_ops(cirq.decompose_once\
                                         ((initial_state(*cirq.LineQubit.range(initial_state.num_qubits())))))

    assert isinstance(circuit1, working_circuit_type) and isinstance(circuit2, working_circuit_type)

def test_State_Prep_incorrect_form_NO_DECOMPOSE_METHOD():
    """
    Test for when state in incorrect form
    NO cirq.decompose used here... therefore should fail with ValueError in __circuit_diagram_info_ stage

    NOTE THAT print statement needed for this!

    """
    initial_state = State_Prep([0,15,1,1])

    with pytest.raises(ValueError) as exc_info:
        assert exc_info is print(cirq.Circuit.from_ops(
            (initial_state(*cirq.LineQubit.range(initial_state.num_qubits())))))

def test_State_Prep_incorrect_form_WITH_DECOMPOSE_METHOD():
    """
    Test for when state in incorrect form
    cirq.decompose used here... therefore should fail with ValueError in _decompose_ stage

    NOTE print statement not needed for this.

    """
    initial_state = State_Prep([0,15,1,1])

    with pytest.raises(ValueError) as exc_info:
        assert exc_info is cirq.Circuit.from_ops(cirq.decompose_once(
            (initial_state(*cirq.LineQubit.range(initial_state.n