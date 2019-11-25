import cirq

# ##circuits_and_constants
# def Simulate_Quantum_Circuit(quantum_circuit_dict, num_shots):
#
#     """
#
#     :param quantum_circuit_dict:
#     :type quantum_circuit_dict: dict
#
#     e.g.
#     {
# 0: {'circuit': None,
#   'factor': (0.10732712612602104+0j),
#   'PauliWord': 'I0 I1 I2 I3'},
#
# 1: {'circuit': 0: ───Rx(0.5π)───@────────────────────────@───Rx(0.5π)───H──────────@────────────────────────@───H────
#                  │                        │                         │                        │
# 1: ──────────────X───@────────────────@───X─────────────────────────X───@────────────────@───X───Rx(0.5π)───@─────────
#                      │                │                                 │                │                  │
# 2: ───X──────────H───X───Rz(1.921π)───X───────H──────────Rx(0.5π)───────X───Rz(1.921π)───X───────Rx(0.5π)───X───@─────
#                                                                                                                 │
# 3: ───X──────────────────────────────────────────────────────────────────────────────────────────H──────────────X───Rz
#   'factor': (0.024523755706991564+0j)
#    'PauliWord': 'Z0 Z1 I2 I3'},
#
#  2: {'circuit': 0: ───Rx(0.5π)───@────────────────────────@───Rx(0.5π)───H──────────@────────────────────────@───H────
#                  │                        │                         │                        │
# 1: ──────────────X───@────────────────@───X─────────────────────────X───@────────────────@───X───Rx(0.5π)───@─────────
#                      │                │                                 │                │                  │
# 2: ───X──────────H───X───Rz(1.921π)───X───────H──────────Rx(0.5π)───────X───Rz(1.921π)───X───────Rx(0.5π)───X───@─────
#                                                                                                                 │
# 3: ───X──────────────────────────────────────────────────────────────────────────────────────────H──────────────X───Rz
#   'factor': (0.011284609976862313+0j)
#   'PauliWord': 'Z0 I1 Z2 I3'},
#  3: {'circuit': 0: ───Rx(0.5π)───@────────────────────────@───Rx(0.5π)───H──────────@────────────────────────@───H────
#                  │                        │                         │                        │
# 1: ──────────────X───@────────────────@───X─────────────────────────X───@────────────────@───X───Rx(0.5π)───@─────────
#                      │                │                                 │                │                  │
# 2: ───X──────────H───X───Rz(1.921π)───X───────H──────────Rx(0.5π)───────X───Rz(1.921π)───X───────Rx(0.5π)───X───@─────
#                                                                                                                 │
# 3: ───X──────────────────────────────────────────────────────────────────────────────────────────H──────────────X───Rz
#   'factor': (0.024157456201338485+0j)
#   'PauliWord': 'Z0 I1 I2 Z3'},
#  4: {'circuit': 0: ───Rx(0.5π)───@────────────────────────@───Rx(0.5π)───H──────────@────────────────────────@───H─────
#                  │                        │                         │                        │
# 1: ──────────────X───@────────────────@───X─────────────────────────X───@────────────────@───X───Rx(0.5π)───@─────────@─
#                      │                │                                 │                │                  │         │
# 2: ───X──────────H───X───Rz(1.921π)───X───────H──────────Rx(0.5π)───────X───Rz(1.921π)───X───────Rx(0.5π)───X───@─────X─
#                                                                                                                 │
# 3: ───X──────────────────────────────────────────────────────────────────────────────────────────H──────────────X───────
#   'factor': (0.024157456201338485+0j)
#   'PauliWord': 'I0 Z1 Z2 I3'},
#
#   ... etc ...
#
#     :return:
#     """
#
#     simulator = cirq.Simulator()
#
#     for key in quantum_circuit_dict:
#
#         if quantum_circuit_dict[key]['circuit'] == None:
#             continue
#         else:
#             circuit = quantum_circuit_dict[key]['circuit']
#             yield simulator.run(circuit, repetitions=num_shots)


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


0: ───Rx(0.5π)───@────────────────────────@───Rx(0.5π)───H──────────@────────────────────────@───H────────────────────
                 │                        │                         │                        │
1: ──────────────X───@────────────────@───X─────────────────────────X───@────────────────@───X───Rx(0.5π)───@─────────
                     │                │                                 │                │                  │
2: ───X──────────H───X───Rz(1.921π)───X───────H──────────Rx(0.5π)───────X───Rz(1.921π)───X───────Rx(0.5π)───X───@─────
                                                                                                                │
3: ───X──────────────────────────────────────────────────────────────────────────────────────────H──────────────X───Rz


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


def Get_parity_of_Binary_counter(binary_counter_result):
    """

    :param binary_counter_result:

    e.g.
         {'10': 485, '00': 515}

    :return:
    e.g.
         {'10': 1, '00': 0}

   NOTE:
        0 = even parity
        1 = odd parity

    """
    Parity_Dic = {}
    for key in binary_counter_result:
        bit_sum = 0
        for bit in key:
            if int(bit) not in [0,1]:
                raise ValueError('state {} not allowed'.format(key))

            bit_sum += int(bit)
            Parity_Dic[key] = bit_sum % 2
    return Parity_Dic


def expectation_value_by_parity(binary_counter_result, Parity_Dic):
    """

    < Z >  = (num_0 - num_1) / total_num_measurements = (num_0 - num_1) / (num_0 + num_1)

    :return:
    """
    num_ones = 0
    num_zeros = 0
    Total = 0
    total_no_measurements = 0
    for state in binary_counter_result:
        if Parity_Dic[state] == 0:
            Total += binary_counter_result[state]
            total_no_measurements += binary_counter_result[state]
        elif Parity_Dic[state] == 1:
            Total -= binary_counter_result[state]
            total_no_measurements += binary_counter_result[state]
        else:
            raise ValueError('state {} not allowed'.format(state))
    #print(binary_counter_result, total_no_measurements, Total)
    expectation_value = Total / total_no_measurements
    return expectation_value


def expectation_value(binary_counter_result):
    """

    < Z >  = (num_0 - num_1) / total_num_measurements = (num_0 - num_1) / (num_0 + num_1)

    :return:
    """
    pass
#     num_ones = 0
#     num_zeros = 0
#     for state in binary_counter_result:
#         individual_bits_in_state = [int(bit) for bit in state]
#         for bit in individual_bits_in_state:
#             if bit == 1:
#                 num_ones += binary_counter_result[state]
#             elif bit == 0:
#                 num_zeros += binary_counter_result[state]
#             else:
#                 raise ValueError('outcomes not in binary {}'.format(individual_bits_in_state))
#     expectation_value = (num_zeros - num_ones) / (num_zeros + num_ones)
#     return expectation_value


class Simulation_Quantum_Circuit_Dict():

    def __init__(self, circuits_factor_PauliWord_dict, num_shots):
        self.circuits_factor_PauliWord_dict = circuits_factor_PauliWord_dict
        self.num_shots = num_shots

        self.hist_key_dict = None
        self.counter_results_raw_dict = None
        self.Identity_result_dict = {}
        self.binary_results_dict = None
        self.parity_results_dict = None
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
                counter_results_raw_dict[key]=  Simulate_Quantum_Circuit(self.circuits_factor_PauliWord_dict[key]['circuit'],
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


    def Get_parity_results_dict(self):
        if self.binary_results_dict == None:
            self.Get_binary_results_dict()

        parity_results_dict = {}
        for key in self.binary_results_dict:
            parity_results_dict[key] = Get_parity_of_Binary_counter(self.binary_results_dict[key])
        self.parity_results_dict = parity_results_dict


    def Get_expectation_value_via_parity(self):
        if self.parity_results_dict == None:
            self.Get_parity_results_dict()

        expect_results_dict = {}
        for key in self.circuits_factor_PauliWord_dict:

            if key in self.Identity_result_dict.keys():
                expect_results_dict[key] = 1

            else:
                expect_results_dict[key] = expectation_value_by_parity(self.binary_results_dict[key], self.parity_results_dict[key])


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

    def Get_expectation_value_dict(self):
        pass
        # if  self.binary_results_dict == None:
        #     self.Get_binary_results_dict()
        #
        # expect_results_dict = {}
        # for key in self.circuits_factor_PauliWord_dict:
        #
        #     if key in self.Identity_result_dict.keys():
        #         expect_results_dict[key] = 1
        #
        #     else:
        #         expect_results_dict[key] = expectation_value(self.binary_results_dict[key])
        #
        #
        # self.expect_results_dict = expect_results_dict

    def Calc_energy(self):
        pass
        # if self.expect_results_dict == None:
        #     self.Get_expectation_value_dict()
        # Energy_list =[]
        # for key in self.circuits_factor_PauliWord_dict:
        #     exp_val = self.expect_results_dict[key]
        #     factor = self.circuits_factor_PauliWord_dict[key]['factor']
        #     Energy_list.append((exp_val*factor))
        #
        # self.Energy = sum(Energy_list)
        #
        # return self.Energy


# xx = Simulation_Quantum_Circuit_Dict(circuits_and_constants, 2000)
# xx.Calc_energy_via_parity()