import cirq

def Get_Histogram_key(PauliWord):
    """

    :param PauliWord:
    :type PauliWord: str

    e.g.
    PauliWord = 'I0 Z1 Z2 I3 I4 X5'


    The code converts to list:
    ['Z0', 'Z1', 'I2', 'I3', 'I4' 'X5']

    and gets non-identity qubit numbers!:
    histogram_string = '0,1,5'


    :return: histogram string
    e.g.
    '0,1,5'
    """
    PauliWord = PauliWord.split(' ')

    string_list = [PauliString[1::] for PauliString in PauliWord if PauliString[0] != 'I']
    histogram_string = ','.join(string_list)

    return histogram_string

def Simulate_Quantum_Circuit(quantum_circuit, num_shots, histogram_string):
    """
    :param num_shots: number of repetitions of Q circuit
    :type num_shots: int


    :param quantum_circuit: Cirq quantum Circuit
    :type quantum_circuit: cirq.circuits.circuit.Circuit

    :param histogram_string: Histogram key string
    :type histogram_string: str


0: ───Rx(0.5π)───@────────────────────────@───Rx(0.5π)───H──────────@────────────────────────@───H──────────────────M
                 │                        │                         │                        │                      │
1: ──────────────X───@────────────────@───X─────────────────────────X───@────────────────@───X───Rx(0.5π)───@───────M
                     │                │                                 │                │                  │       │
2: ───X──────────H───X───Rz(1.921π)───X───────H──────────Rx(0.5π)───────X───Rz(1.921π)───X───────Rx(0.5π)───X───@───M
                                                                                                                │   │
3: ───X──────────────────────────────────────────────────────────────────────────────────────────H──────────────X───M


    :return: Return counter result
    :rtype: collections.Counter

    e.g.
    Counter({1: 654, 0: 346})
    """

    simulator = cirq.Simulator()
    raw_result = simulator.run(quantum_circuit, repetitions=num_shots)
    hist_result = raw_result.histogram(key=histogram_string)

    return hist_result

def Return_as_binary(counter_result, PauliWord):
    """
    Takes in counter_result and gives result with keys as quantum states (binary)

    :param counter_result: histogram counter results from quantum circuit simulation
    :type counter_result: collections.Counter

    e.g.
    Counter({2: 485, 1: 515})

    :return:
    e.g.
    {'10': 485, '01': 515}
    """
    state_dictionary ={}

    PauliWord = PauliWord.split(' ')
    num_terms_to_measure = len([i for i in PauliWord if i[0] != 'I'])
    binary_length = '{' + '0:0{}b'.format(num_terms_to_measure) + '}'

    for output_state in counter_result:
        binary_key = binary_length.format(output_state)
        state_dictionary[binary_key] = counter_result[output_state]

    return state_dictionary

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
        if self.hist_key_dict == None:
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
        if self.counter_results_raw_dict == None:
            self.Get_counter_results_dict()

        binary_results_dict = {}
        for key in self.counter_results_raw_dict:
            binary_results_dict[key] = Return_as_binary(self.counter_results_raw_dict[key],
                                                         self.circuits_factor_PauliWord_dict[key]['PauliWord'])
        self.binary_results_dict = binary_results_dict

    def Get_expectation_value_via_parity(self):
        if self.binary_results_dict == None:
            self.Get_binary_results_dict()

        expect_results_dict = {}
        for key in self.circuits_factor_PauliWord_dict:

            if key in self.Identity_result_dict.keys():
                expect_results_dict[key] = 1

            else:
                expect_results_dict[key] = expectation_value_by_parity(self.binary_results_dict[key])


        self.expect_results_dict = expect_results_dict

    def Calc_energy_via_parity(self):
        if self.expect_results_dict == None:
            self.Get_expectation_value_via_parity()

        Energy_list =[]
        for key in self.circuits_factor_PauliWord_dict:
            exp_val = self.expect_results_dict[key]
            factor = self.circuits_factor_PauliWord_dict[key]['gamma_l']
            Energy_list.append((exp_val*factor))

        self.Energy = sum(Energy_list)

        return self.Energy
