import cirq
import numpy as np
from quchem.Qcircuit.Hamiltonian_term_measurement_functions import change_pauliword_to_Z_basis_then_measure

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

### Build full Q circuit of Ansatz + measurement of Pauli in Z basis
def Generate_Ansatz_and_PauliMeasurement_Q_Circuit(Full_Ansatz_Q_Circuit, PauliWord_QubitOp):
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

def Generate_Ansatz_and_PauliMeasurement_Q_Circuit_of_Molecular_Hamiltonian(Full_Ansatz_Q_Circuit, QubitOperator_Hamiltonian):
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
                temp_d['circuit'] = Generate_Ansatz_and_PauliMeasurement_Q_Circuit(Full_Ansatz_Q_Circuit, PauliWord_QubitOp)
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

