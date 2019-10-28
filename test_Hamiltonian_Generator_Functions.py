from Hamiltonian_Generator_Functions import Hamiltonian
import pytest



def test_geometry():
    """
    Standard use test
    """

    X = Hamiltonian('H2')

    X.Get_Qubit_Hamiltonian()


    working_circuit_type = type(cirq.Circuit.from_ops(cirq.H(1)))
    initial_state = State_Prep([0, 0, 1, 1])
    circuit1 = cirq.Circuit.from_ops((initial_state(*cirq.LineQubit.range(initial_state.num_qubits()))))
    circuit2 = cirq.Circuit.from_ops(cirq.decompose_once\
                                         ((initial_state(*cirq.LineQubit.range(initial_state.num_qubits())))))

    assert isinstance(circuit1, working_circuit_type) and isinstance(circuit2, working_circuit_type)