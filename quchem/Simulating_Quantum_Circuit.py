import cirq
import numpy as np

def Get_Histogram_key(qubitOperator):
    """
     Function to obtain histogram key string for Cirq Simulator.

    e.g.
            PauliWord = QubitOperator('X0 Z2 Y3', 0.5j)

            returning: histogram_string = '0,2,3'


    Args:
        qubitOperator (openfermion.ops._qubit_operator.QubitOperator): QubitOperator

    Returns:
        histogram_string (str): Returns string corresponding to histogram key (required for Cirq simulator)

    """

    qubit_No, PauliStr = zip(*list(*qubitOperator.terms.keys()))
    histogram_string = ','.join([str(i) for i in qubit_No])
    return histogram_string

def Simulate_Quantum_Circuit(quantum_circuit, num_shots, histogram_string):
    """
     Function to simulate quantum circuit and give counter corresponding to number of times state measured.

    e.g.
            quantum_circuit =
                                0: ───────@──────────────────────────────@───Z──────────────────────M───
                                          │                              │                          │
                                1: ───H───X──────────@───────────────@───X───H───────────X───H──────M───
                                                     │               │                              │
                                2: ───X──────────────┼───────────────┼──────────────────────────────┼───
                                                     │               │                              │
                                3: ───X───Rx(0.5π)───X───Rz(-0.5π)───X───────Rx(-0.5π)───Y───S^-1───M───

            histogram_string = '0,1,3'
            num_shots = 10000

        Simulate_Quantum_Circuit(quantum_circuit, num_shots, histogram_string)
        >> Counter({3: 2617, 2: 2365, 1: 2420, 0: 2598})


    Args:
        quantum_circuit (cirq.circuits.circuit.Circuit): Cirq Quantum circuit to simulate.
        num_shots (int): Number of times to repeat simlation.
        histogram_string (str): Histrogram key string (Corresponds to to string of qubit numbers to take results from)

    Returns:
        hist_result (collections.Counter): Return counter result dict with entries of (state: num times obtained)

    """

    simulator = cirq.Simulator()
    raw_result = simulator.run(quantum_circuit, repetitions=num_shots)
    hist_result = raw_result.histogram(key=histogram_string)

    return hist_result

def Get_wavefunction(quantum_circuit, sig_figs=3):
    """
     Function to simulate quantum circuit and wavefunction

    """
    quantum_circuit_M_gates_removed = quantum_circuit.moments[
                                      :-1]  # removes last moment (aka measurement step - which collapses wavefunction)
    quantum_circuit_new = cirq.Circuit(quantum_circuit_M_gates_removed)
    simulator = cirq.Simulator()
    result = simulator.simulate(quantum_circuit_new, qubit_order=quantum_circuit_new.all_qubits())
    print(np.around(result.final_state, sig_figs))
    return result.final_state

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

def Return_as_binary(counter_result, histogram_string):
    """
     Function to convert counter_result into counter with keys as quantum states (binary)

    e.g.
           counter_result =  Counter({3: 2617, 2: 2365, 1: 2420, 0: 2598})
           PauliWord = 'Z0 X1 I2 Y3'

           Return_as_binary(counter_result, PauliWord)
           >> {'011': 2617, '010': 2365, '001': 2420, '000': 2598}

    Args:
        counter_result (collections.Counter): Counter result dict with entries of (state: num times obtained)
        PauliWord (str): PauliWord measured as string

    Returns:
        state_dictionary (dict): Return counter result dict with states in binary form

    """
    n_qubits_measured = len(histogram_string.split(','))

    counter_result_binary = {}
    for int_state in counter_result:
        binary_state = Get_state_as_str(n_qubits_measured, int_state)

        counter_result_binary[binary_state] = counter_result[int_state]

    return counter_result_binary

def calc_parity(state):
    """
    Takes in a state and returns its parity (even = 0 , odd = 1)

    :param state:
    :type state: str

    :return: parity of state
    :rtype: int
    """
    bit_sum = 0
    for bit in state:
        if int(bit) not in [0,1]:
            raise ValueError('state {} not allowed'.format(state))
        bit_sum += int(bit)
    parity = bit_sum % 2
    return parity

def expectation_value_by_parity(binary_counter_result):
    """

    < Z >  = (num_0 - num_1) / total_num_measurements = (num_0 - num_1) / (num_0 + num_1)

    note that for multiple qubits one multiplies <Z> on each line. Therefore can calculate value from parity
    of output bit string

    :param binary_counter_result:
    :type binary_counter_result: dict
    e.g.
        {
            1: {'11': 10000},
            2: {'10': 9998, '01': 2},
            3: {'10': 10000},
            4: {'10': 10000},
            5: {'10': 10000},
            6: {'00': 9995, '01': 4, '10': 1},
            7: {'1': 9334, '0': 666},
            8: {'1': 9351, '0': 649},
            9: {'0': 9606, '1': 394},
            10: {'0': 9594, '1': 406}
        }

    :return: expectation value <Z>
    :rtype: float
    """
    Total = 0
    total_no_measurements = 0
    for state in binary_counter_result:
        parity = calc_parity(state)
        if parity == 0:
            Total += binary_counter_result[state]
            total_no_measurements += binary_counter_result[state]
        elif parity == 1:
            Total -= binary_counter_result[state]
            total_no_measurements += binary_counter_result[state]
        else:
            raise ValueError('state {} not allowed'.format(state))
    #print(binary_counter_result, total_no_measurements, Total)
    expectation_value = Total / total_no_measurements
    return expectation_value

class Simulation_Quantum_Circuit_Dict():

    def __init__(self, circuits_factor_PauliWord_dict, num_shots):
        self.circuits_factor_PauliWord_dict = circuits_factor_PauliWord_dict
        self.num_shots = num_shots

        self.hist_key_dict = None
        self.counter_results_raw_dict = None
        self.Identity_result_dict = {}
        self.binary_results_dict = None
        self.expect_results_dict = None

    def Get_Histkey_dict(self):
        hist_key_dict={}
        for key in self.circuits_factor_PauliWord_dict:
            hist_key_dict[key]= Get_Histogram_key(self.circuits_factor_PauliWord_dict[key]['PauliWord'])
        self.hist_key_dict = hist_key_dict

    def Get_counter_results_dict(self):
        if self.hist_key_dict is None:
            self.Get_Histkey_dict()

        counter_results_raw_dict = {}
        for key in self.circuits_factor_PauliWord_dict:

            if self.hist_key_dict[key] != '':
                #checks for non identity ciruict
                counter_results_raw_dict[key] = Simulate_Quantum_Circuit(self.circuits_factor_PauliWord_dict[key]['circuit'],
                                                            self.num_shots, self.hist_key_dict[key])
            else:
                self.Identity_result_dict[key]= (self.circuits_factor_PauliWord_dict[key]['PauliWord'], self.circuits_factor_PauliWord_dict[key]['gamma_l'])

        self.counter_results_raw_dict = counter_results_raw_dict

    def Get_binary_results_dict(self):
        if self.counter_results_raw_dict is None:
            self.Get_counter_results_dict()

        binary_results_dict = {}
        for key in self.counter_results_raw_dict:
            binary_results_dict[key] = Return_as_binary(self.counter_results_raw_dict[key],
                                                         self.circuits_factor_PauliWord_dict[key]['PauliWord'])
        self.binary_results_dict = binary_results_dict

    def Get_expectation_value_via_parity(self):
        if self.binary_results_dict is None:
            self.Get_binary_results_dict()

        expect_results_dict = {}
        for key in self.circuits_factor_PauliWord_dict:

            if key in self.Identity_result_dict.keys():
                expect_results_dict[key] = 1

            else:
                expect_results_dict[key] = expectation_value_by_parity(self.binary_results_dict[key])


        self.expect_results_dict = expect_results_dict

    def Calc_energy_via_parity(self):
        if self.expect_results_dict is None:
            self.Get_expectation_value_via_parity()

        Energy_list =[]
        for key in self.circuits_factor_PauliWord_dict:
            exp_val = self.expect_results_dict[key]
            factor = self.circuits_factor_PauliWord_dict[key]['gamma_l']
            Energy_list.append((exp_val*factor))

        self.Energy_list = Energy_list
        self.Energy = sum(Energy_list)

        return self.Energy

class Simulate_Single_Circuit():

    def __init__(self, PauliWord, quantum_circuit, num_shots):
        self.PauliWord = PauliWord
        self.quantum_circuit = quantum_circuit
        self.num_shots = num_shots

        self.histogram_string = None
        self.counter_results_raw = None
        self.binary_results = None
        self.expect_result = None

    def Get_Histkey_method(self):
        self.histogram_string = Get_Histogram_key(self.PauliWord)

    def Get_counter_results_method(self):
        if self.histogram_string is None:
            self.Get_Histkey_method()

        self.counter_results_raw = Simulate_Quantum_Circuit(self.quantum_circuit, self.num_shots, self.histogram_string)


    def Get_binary_results_method(self):
        if self.counter_results_raw is None:
            self.Get_counter_results_method()

        self.binary_results = Return_as_binary(self.counter_results_raw, self.PauliWord)

    def Get_expectation_value_via_parity(self):
        if self.binary_results is None:
            self.Get_binary_results_method()

        expect_result = expectation_value_by_parity(self.binary_results)


        self.expect_result = expect_result
        return expect_result


from quchem.quantum_circuit_functions import Generate_Full_Q_Circuit
class VQE_Experiment():
    def __init__(self, qubitHamiltonian, ansatz_circuit, n_shots):
        self.qubitHamiltonian = qubitHamiltonian
        self.ansatz_circuit = ansatz_circuit
        self.n_shots = n_shots

    def Calc_Energy(self):

        E_list=[]
        for qubitOp in self.qubitHamiltonian:
            for PauliWord, const in qubitOp.terms.items():
                if PauliWord is not ():
                    Q_circuit = Generate_Full_Q_Circuit(self.ansatz_circuit, qubitOp)
                    hist_key_str = Get_Histogram_key(qubitOp)
                    int_state_counter = Simulate_Quantum_Circuit(Q_circuit, self.n_shots, hist_key_str)
                    binary_state_counter = Return_as_binary(int_state_counter, hist_key_str)
                    exp_result = expectation_value_by_parity(binary_state_counter)
                    E_list.append(exp_result*const)
                else:
                    E_list.append(const)
        return sum(E_list).real

    def Get_wavefunction_of_state(self, sig_figs=3):
        return Get_wavefunction(self.ansatz_circuit, sig_figs=sig_figs)

