# https://arxiv.org/pdf/quant-ph/0104030.pdf
# ^^^ Need to be able to prepare arbitrary state!
import numpy as np
import cirq

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

# state = |000> + |001> + |010> + |011> + |100> + |101 > + |110 > + |111>



class My_U_Gate(cirq.SingleQubitGate):
    """
    Description

    Args:
        theta (float): angle to rotate by in radians.
        number_control_qubits (int): number of control qubits
    """

    def __init__(self, theta, number_control_qubits):
        self.theta = theta
        self.num_control_qubits = number_control_qubits
    def _unitary_(self):
        Unitary_Matrix = np.array([
                    [np.cos(self.theta), np.sin(self.theta)],
                    [np.sin(self.theta), -1* np.cos(self.theta)]
                ])
        return Unitary_Matrix
    def num_qubits(self):
        return 1
    # def _circuit_diagram_info_(self, args):
    #     return 'U = {} rad'

    def _circuit_diagram_info_(self,args):
        return cirq.CircuitDiagramInfo(
            wire_symbols=tuple([*['@' for _ in range(self.num_control_qubits-1)],' U = {} rad '.format(self.theta.__round__(4))]),exponent=1)

if __name__ == '__main__':
    theta = np.pi

    U_single_qubit = My_U_Gate(theta,0)
    op = U_single_qubit.on(cirq.LineQubit(1))
    print(cirq.Circuit.from_ops(op))

    # cirq.Gate.controlled().on(control_qubit, target_qubit)
    op = U_single_qubit.controlled(num_controls=3, control_values=[0, 0, 1]).on(
        *[cirq.LineQubit(1), cirq.LineQubit(2), cirq.LineQubit(3)], cirq.LineQubit(4))
    print(cirq.Circuit(op))


class State_Prep_Circuit(cirq.Gate):
    """
    Function to build cirq Circuit that will make an arbitrary state!

    e.g.:
   {
        (1, 0): 0.5092156980522868,
        (2, 0): 0.9097461710606383,
        (2, 1): 0.9338960671361634,
        (3, 0): 0.3007458481278772,
        (3, 1): 0.5945638342986989,
        (3, 2): 0.7174996992546281,
        (3, 3): 0.7105908988639925
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
    def __init__(self, circuit_param_dict):

        self.circuit_param_dict = circuit_param_dict

    def _decompose_(self, qubits):

        for Tuple in self.circuit_param_dict:

            theta = self.circuit_param_dict[Tuple]
            U_single_qubit = My_U_Gate(theta, 0)

            if Tuple[0]=='': #Tuple[0]==1:
                num_controls = 0
                control_values = []
            else:
                num_controls = Tuple[0] #Tuple[0] - 1
                control_values=[int(bit) for bit in Tuple[1]]

            # qubit_list = cirq.LineQubit.range(0,Tuple[0])
            qubit_list = qubits[0:Tuple[0]+1]
            yield U_single_qubit.controlled(num_controls=num_controls, control_values=control_values).on(*qubit_list)

    def _circuit_diagram_info_(self, args):

        max_qubit = max(Tuple[0] for Tuple in alpha_j)
        string_list = []
        for i in range(max_qubit):
            string_list.append('state prep circuit')
        return string_list

    def num_qubits(self):
        max_qubit = max(Tuple[0]+1 for Tuple in alpha_j) # +1 due to python indexing
        return max_qubit

# if __name__ == '__main__':
#     num_qub = 3
#     Coefficient_list=None
#     alpha_j = Get_state_prep_dict(num_qub, Coefficient_list=Coefficient_list)
#     state_circ = State_Prep_Circuit(alpha_j)
#     print(cirq.Circuit(state_circ(*cirq.LineQubit.range(state_circ.num_qubits()))))
#     print(
#         cirq.Circuit(cirq.decompose_once((state_circ(*cirq.LineQubit.range(state_circ.num_qubits()))))))


def Get_state_prep_dict(num_qubits, Coefficient_list=None):

    if Coefficient_list is None:
        Coefficient_list= np.random.rand(2 ** num_qubits)

    state_list = [Get_state_as_str(num_qubits, i) for i in range(2 ** num_qubits)]
    alpha_j = {}
    for qubit_j in range(num_qubits):
        state_toJneg1 = set([state[:qubit_j] for state in state_list])

        for term in state_toJneg1:
            upper_term = term + '1'
            lower_term = term + '0'

            upper_sum = []
            lower_sum = []

            for k in range(len(state_list)):
                state = state_list[k]
                if state[:qubit_j + 1] == upper_term:
                    upper_sum.append(Coefficient_list[k] ** 2)
                elif state[:qubit_j + 1] == lower_term:
                    lower_sum.append(Coefficient_list[k] ** 2)
            alpha_j[(qubit_j, term)] = np.arctan(np.sqrt(sum(upper_sum) / sum(lower_sum)))
    return alpha_j

if __name__ == '__main__':
    num_qub = 3
    # Coefficient_list = np.random.rand(2 ** num_qub)
    Coefficient_list = [np.sqrt(0.3), np.sqrt(0.1), np.sqrt(0.1), np.sqrt(0.1), np.sqrt(0.1), np.sqrt(0.1),
                        np.sqrt(0.1), np.sqrt(0.1)]  # [1/2,1/2,1/2,1/2]  #[0.9, 0.3, 0.3, 0.1]

    alpha_j = Get_state_prep_dict(num_qub, Coefficient_list=Coefficient_list)
    state_circ = State_Prep_Circuit(alpha_j)
    circuit = (cirq.Circuit(cirq.decompose_once((state_circ(*cirq.LineQubit.range(state_circ.num_qubits()))))))

    # MEASURE
    qubits_to_measure = (cirq.LineQubit(q_No) for q_No in range(num_qub))
    circuit.append(cirq.measure(*qubits_to_measure))
    print(circuit)

    # simulate
    simulator = cirq.Simulator()
    results = simulator.run(circuit, repetitions=100000)
    print(results.histogram(key='0,1,2'))
    print('actual state:')
    # NOTE! must not have any measurement (otherwise collapses state!)
    state_circ = State_Prep_Circuit(alpha_j)
    circuit = (cirq.Circuit(cirq.decompose_once((state_circ(*cirq.LineQubit.range(state_circ.num_qubits()))))))
    qubits_to_measure = (cirq.LineQubit(q_No) for q_No in range(num_qub))
    result = simulator.simulate(circuit, qubit_order=qubits_to_measure)
    print(np.around(result.final_state, 3))
    print('expected state amplitudes:', Coefficient_list)

    # alternative key method! key fined by ## HERE ###
    # # MEASURE
    # qubits_to_measure = (cirq.LineQubit(q_No) for q_No in range(num_qub))
    # circuit.append(cirq.measure(*qubits_to_measure, key='Z'))             ## HERE ### (can be any str)
    # print(circuit)
    #
    # # simulate
    # simulator = cirq.Simulator()
    # results = simulator.run(circuit, repetitions=1000)
    # print(results.histogram(key='Z'))                         ## re-use ###