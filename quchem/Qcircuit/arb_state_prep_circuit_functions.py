import cirq
import numpy as np
from quchem.Qcircuit.misc_quantum_circuit_functions import Get_state_as_str


class My_U_Gate(cirq.SingleQubitGate):
    """
    U_theta single qubit gate defined in: https://arxiv.org/pdf/quant-ph/0104030.pdf

    Upon operation, a qubitin the state|0〉is transformed into a superposition inthe two state:  (cosθ,sinθ).
    Similarly, a qubit in thestate|1〉is transformed into (sinθ,−cosθ).

    for special angles U_theta reduces to certain gates:
        - theta = pi/4 reduces to the Hadamard gate!
        - theta = pi/2 reduces to the X gate!

    Args:
        theta (float): angle to rotate by in radians.
        number_control_qubits (int): number of control qubits
    """

    def __init__(self, theta):
        self.theta = theta
    def _unitary_(self):
        Unitary_Matrix = np.array([
                    [np.cos(self.theta), np.sin(self.theta)],
                    [np.sin(self.theta), -1* np.cos(self.theta)]
                ])
        return Unitary_Matrix
    def num_qubits(self):
        return 1

    def _circuit_diagram_info_(self,args):
        # return cirq.CircuitDiagramInfo(
        #     wire_symbols=tuple([*['@' for _ in range(self.num_control_qubits-1)],' U = {} rad '.format(self.theta.__round__(4))]),exponent=1)
        return ' U = {} rad '.format(self.theta.__round__(4))

    def __str__(self):
        return ' U = {} rad '.format(self.theta.__round__(4))

    def __repr__(self):
        return ' U_arb_state_prep'

def Get_arb_state_prep_circuit_params(N_qubits, state_vector):
    
    if len(state_vector) != 2 ** N_qubits:
        raise ValueError('incorrect number of coefficients')
    
    if not np.isclose(sum(np.abs(state_vector)**2), 1):
        raise ValueError('state_vector is not normalized') 

    alpha_j_dict = {}
    if N_qubits==1:
        theta = np.arccos(state_vector[0])
        alpha_j_dict[target_qubit_index] = [{'control_state': None, 'angle': theta}]
        return alpha_j_dict
    else:
        for target_qubit_index in range(N_qubits-1):
            number_control_qubits = target_qubit_index

            if number_control_qubits==0:
                # no controls required for 0th qubit!
                control_state_list = ['']
            else:
                control_state_list = [Get_state_as_str(number_control_qubits, i) for i in range(2 ** number_control_qubits)]

            operation_list=[]
            for control_state in control_state_list:
                numerator_control_str = control_state + '1'
                denominator_control_str = control_state + '0'

                numerator=0
                denominator=0
                for vector_amp_index, amplitude in enumerate(state_vector):
                    binary_state = Get_state_as_str(N_qubits, vector_amp_index)

                    if binary_state[:target_qubit_index + 1] == numerator_control_str:
                        numerator+=np.abs(amplitude)**2

                    if binary_state[:target_qubit_index + 1] == denominator_control_str:
                        denominator+=np.abs(amplitude)**2
                ##
                if (numerator == 0) and (denominator == 0):
                    angle = 0
                else:
                    angle = np.arctan(np.sqrt(numerator / denominator))

                operation_list.append({'control_state': control_state, 'angle': angle})
            alpha_j_dict[target_qubit_index]= operation_list

            ### final rotation

            final_qubit_index = N_qubits-1
            operation_list=[] 
            control_state_list = [Get_state_as_str(final_qubit_index, i) for i in range(2 ** final_qubit_index)]
            for control_state in control_state_list:
                numerator_control_str = control_state + '1'
                denominator_control_str = control_state + '0'

                numerator= state_vector[int(numerator_control_str,2)]
                denominator= state_vector[int(denominator_control_str,2)]

                if (numerator == 0) and (denominator == 0):
                    angle = 0
                else:
                    angle = np.arctan(np.sqrt(numerator / denominator))

                operation_list.append({'control_state': control_state, 'angle': angle})
            alpha_j_dict[final_qubit_index]= operation_list
        return alpha_j_dict

class State_Prep_Circuit(cirq.Gate):
    """
    Function to build cirq Circuit that will make an arbitrary state!

    e.g.:
    {
         0: [{'control_state': None, 'angle': 0.7853981633974483}],
         1: [{'control_state': '0', 'angle': 0.7853981633974483},
          {'control_state': '1', 'angle': 0.7853981633974483}]
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

    def __init__(self, circuit_param_dict, N_system_qubits=0):

        self.circuit_param_dict = circuit_param_dict
        self.N_system_qubits = N_system_qubits

    def _decompose_(self, qubits):

        for qubit in sorted(self.circuit_param_dict.keys()):

            for term in self.circuit_param_dict[qubit]:
                if term['control_state']:
                    control_values = [int(bit) for bit in term['control_state']]
                    num_controls = len(control_values)
                    theta = term['angle']

                    if theta == 0:
                        # yield cirq.I.on(cirq.LineQubit(qubit+self.N_system_qubits))
                        pass
                    elif np.isclose(theta, np.pi/4):
                        #theta = pi/4 reduces to the Hadamard gate!
                        qubit_list = cirq.LineQubit.range(self.N_system_qubits, self.N_system_qubits+1 + num_controls)
                        yield cirq.H.controlled(num_controls=num_controls, control_values=control_values).on(
                            *qubit_list)
                    elif np.isclose(theta, np.pi/2):
                        #theta = pi/2 reduces to the X gate!
                        qubit_list = cirq.LineQubit.range(self.N_system_qubits, self.N_system_qubits+1 + num_controls)
                        yield cirq.X.controlled(num_controls=num_controls, control_values=control_values).on(
                            *qubit_list)
                    else:
                        U_single_qubit = My_U_Gate(theta)
                        qubit_list = cirq.LineQubit.range(self.N_system_qubits, self.N_system_qubits+1 + num_controls)
                        yield U_single_qubit.controlled(num_controls=num_controls, control_values=control_values).on(
                            *qubit_list)
                else:
                    theta = term['angle']
                    if theta == 0:
                        yield cirq.I.on(cirq.LineQubit(qubit+self.N_system_qubits))
                    elif np.isclose(theta, np.pi/4):
                        yield cirq.H.on(cirq.LineQubit(qubit+self.N_system_qubits))
                    elif np.isclose(theta, np.pi/2):
                        yield cirq.X.on(cirq.LineQubit(qubit+self.N_system_qubits))
                    else:
                        yield My_U_Gate(theta).on(cirq.LineQubit(qubit+self.N_system_qubits))
    def _circuit_diagram_info_(self, args):

        max_qubit = max(self.circuit_param_dict.keys())
        string_list = []
        for i in range(max_qubit):
            string_list.append('state prep circuit')
        return string_list

    def num_qubits(self):
        return max(self.circuit_param_dict.keys())

class State_Prep_Circuit_DAGGER(cirq.Gate):
    """
    dagger of State_Prep_Circuit function.

    Args:
        circuit_param_dict (dict): A Dictionary of Tuples (qubit, control_val(int)) value is angle

    Returns
        A cirq circuit object to be used by cirq.Circuit.from_ops to generate arbitrary state

    """

    def __init__(self, circuit_param_dict, N_system_qubits=0):

        self.circuit_param_dict = circuit_param_dict
        self.N_system_qubits = N_system_qubits

    def _decompose_(self, qubits):

        for qubit in sorted(self.circuit_param_dict.keys())[::-1]: #reversed!

            for term in self.circuit_param_dict[qubit][::-1]: #reversed!
                if term['control_state']:
                    control_values = [int(bit) for bit in term['control_state']]
                    num_controls = len(control_values)
                    theta = term['angle'] 

                    if theta == 0:
                        # yield cirq.I.on(cirq.LineQubit(qubit+self.N_system_qubits))
                        pass
                    elif np.isclose(theta, np.pi/4):
                        #theta = pi/4 reduces to the Hadamard gate!
                        qubit_list = cirq.LineQubit.range(self.N_system_qubits, self.N_system_qubits+1 + num_controls)
                        yield cirq.H.controlled(num_controls=num_controls, control_values=control_values).on(
                            *qubit_list)
                    elif np.isclose(theta, np.pi/2):
                        #theta = pi/2 reduces to the X gate!
                        qubit_list = cirq.LineQubit.range(self.N_system_qubits, self.N_system_qubits+1 + num_controls)
                        yield cirq.X.controlled(num_controls=num_controls, control_values=control_values).on(
                            *qubit_list)
                    else:
                        U_single_qubit = My_U_Gate(theta*-1) #negative for dagger!
                        qubit_list = cirq.LineQubit.range(self.N_system_qubits, self.N_system_qubits+1 + num_controls)
                        yield U_single_qubit.controlled(num_controls=num_controls, control_values=control_values).on(
                            *qubit_list)
                else:
                    theta = term['angle']
                    if theta == 0:
                        yield cirq.I.on(cirq.LineQubit(qubit+self.N_system_qubits))
                    elif np.isclose(theta, np.pi/4):
                        yield cirq.H.on(cirq.LineQubit(qubit+self.N_system_qubits))
                    elif np.isclose(theta, np.pi/2):
                        yield cirq.X.on(cirq.LineQubit(qubit+self.N_system_qubits))
                    else:
                        yield My_U_Gate(theta*-1).on(cirq.LineQubit(qubit+self.N_system_qubits)) #negative for dagger!
    def _circuit_diagram_info_(self, args):

        max_qubit = max(self.circuit_param_dict.keys())
        string_list = []
        for i in range(max_qubit):
            string_list.append('state prep circuit')
        return string_list

    def num_qubits(self):
        return max(self.circuit_param_dict.keys())

class Get_G_and_Gdag_circuits():
    
    def __init__(self, ancilla_state, N_system_qubits, check_state_prep_circuit=False):
        self.ancilla_state=ancilla_state
        self.N_system_qubits = N_system_qubits
        self.N_ancilla = int(np.ceil(np.log2(len(ancilla_state))))
        
        self.G_circuit = None
        self.G_DAGGER_circuit=None
        self.check_circuit = check_state_prep_circuit
        self._get_circuits()
        
    def _get_circuits(self):
        alpha_j_dict = Get_arb_state_prep_circuit_params(self.N_ancilla, self.ancilla_state)
        
        G_circ_obj = State_Prep_Circuit(alpha_j_dict, N_system_qubits=self.N_system_qubits)
        self.G_circuit = (
                cirq.Circuit(cirq.decompose_once((G_circ_obj(*cirq.LineQubit.range(self.N_system_qubits,
                 self.N_system_qubits+G_circ_obj.num_qubits()))))))
        
        G_DAGGER_obj = State_Prep_Circuit_DAGGER(alpha_j_dict, N_system_qubits=self.N_system_qubits)
        self.G_DAGGER_circuit = (
                cirq.Circuit(cirq.decompose_once((G_DAGGER_obj(*cirq.LineQubit.range(self.N_system_qubits,
                    self.N_system_qubits+G_DAGGER_obj.num_qubits()))))))
        
        if self.check_circuit:
            G_mat = self.G_circuit.unitary()
            if not np.allclose(G_mat[:,0], self.ancilla_state):
                raise ValueError('Incorrect state being prepared!')
                
            if not np.allclose(G_mat.dot(self.G_DAGGER_circuit.unitary().conj().T), np.eye(2**self.N_ancilla)):
                raise ValueError('G not unitary')