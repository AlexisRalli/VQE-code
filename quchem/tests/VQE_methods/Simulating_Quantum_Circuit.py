import cirq

###circuits_and_constants

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


def Get_Histogram_key(quantum_circuit_dict):
    """

    :param quantum_circuit_dict:

    Takes PauliWord:
    ['Z0', 'Z1', 'I2', 'I3', X5]

    gets non-identity qubit numbers!:
    histogram_string = '0,1,5'


    :return:
    e.g.
    {0: '',
     1: '0,1',
     2: '0,2',
     3: '0,3',
     4: '1,2',
     5: '1,3',
     6: '2,3',
     7: '0',
     8: '1',
     9: '2',
     10: '3'}

    """
    hist_key_dict={}
    for key in quantum_circuit_dict:
        PauliWord = quantum_circuit_dict[key]['PauliWord']
        PauliWord = PauliWord.split(' ')

        string_list = [PauliString[1::] for PauliString in PauliWord if PauliString[0] != 'I']
        histogram_string = ','.join(string_list)
        hist_key_dict[key] = histogram_string
    return hist_key_dict


def Simulate_Quantum_Circuit(quantum_circuit_dict, num_shots):
    """
    :param num_shots: number of repetitions of Q circuit
    :type num_shots: int


    :param quantum_circuit_dict:
    :type quantum_circuit_dict: dict

    e.g.
    {
0: {'circuit': None,
  'factor': (0.10732712612602104+0j),
  'PauliWord': 'I0 I1 I2 I3'},

1: {'circuit': 0: ───Rx(0.5π)───@────────────────────────@───Rx(0.5π)───H──────────@────────────────────────@───H────
                 │                        │                         │                        │
1: ──────────────X───@────────────────@───X─────────────────────────X───@────────────────@───X───Rx(0.5π)───@─────────
                     │                │                                 │                │                  │
2: ───X──────────H───X───Rz(1.921π)───X───────H──────────Rx(0.5π)───────X───Rz(1.921π)───X───────Rx(0.5π)───X───@─────
                                                                                                                │
3: ───X──────────────────────────────────────────────────────────────────────────────────────────H──────────────X───Rz
  'factor': (0.024523755706991564+0j)
   'PauliWord': 'Z0 Z1 I2 I3'},

 2: {'circuit': 0: ───Rx(0.5π)───@────────────────────────@───Rx(0.5π)───H──────────@────────────────────────@───H────
                 │                        │                         │                        │
1: ──────────────X───@────────────────@───X─────────────────────────X───@────────────────@───X───Rx(0.5π)───@─────────
                     │                │                                 │                │                  │
2: ───X──────────H───X───Rz(1.921π)───X───────H──────────Rx(0.5π)───────X───Rz(1.921π)───X───────Rx(0.5π)───X───@─────
                                                                                                                │
3: ───X──────────────────────────────────────────────────────────────────────────────────────────H──────────────X───Rz
  'factor': (0.011284609976862313+0j)
  'PauliWord': 'Z0 I1 Z2 I3'},
 3: {'circuit': 0: ───Rx(0.5π)───@────────────────────────@───Rx(0.5π)───H──────────@────────────────────────@───H────
                 │                        │                         │                        │
1: ──────────────X───@────────────────@───X─────────────────────────X───@────────────────@───X───Rx(0.5π)───@─────────
                     │                │                                 │                │                  │
2: ───X──────────H───X───Rz(1.921π)───X───────H──────────Rx(0.5π)───────X───Rz(1.921π)───X───────Rx(0.5π)───X───@─────
                                                                                                                │
3: ───X──────────────────────────────────────────────────────────────────────────────────────────H──────────────X───Rz
  'factor': (0.024157456201338485+0j)
  'PauliWord': 'Z0 I1 I2 Z3'},
 4: {'circuit': 0: ───Rx(0.5π)───@────────────────────────@───Rx(0.5π)───H──────────@────────────────────────@───H─────
                 │                        │                         │                        │
1: ──────────────X───@────────────────@───X─────────────────────────X───@────────────────@───X───Rx(0.5π)───@─────────@─
                     │                │                                 │                │                  │         │
2: ───X──────────H───X───Rz(1.921π)───X───────H──────────Rx(0.5π)───────X───Rz(1.921π)───X───────Rx(0.5π)───X───@─────X─
                                                                                                                │
3: ───X──────────────────────────────────────────────────────────────────────────────────────────H──────────────X───────
  'factor': (0.024157456201338485+0j)
  'PauliWord': 'I0 Z1 Z2 I3'},

  ... etc ...

    :return:
    :rtype: dict

    e.g.
    {1: Counter({3: 795, 0: 205}),
     2: Counter({2: 796, 1: 204}),
     3: Counter({1: 194, 2: 806}),
     4: Counter({2: 778, 1: 222}),
     5: Counter({2: 794, 1: 206}),
     6: Counter({3: 212, 0: 788}),
     7: Counter({1: 915, 0: 85}),
     8: Counter({0: 67, 1: 933}),
     9: Counter({0: 909, 1: 91}),
     10: Counter({1: 639, 0: 361})}

    """
    histogram_strings = Get_Histogram_key(quantum_circuit_dict)
    #raw_results = Simulate_Quantum_Circuit(quantum_circuit_dict, num_shots)

    simulator = cirq.Simulator()
    results={}
    for key in quantum_circuit_dict:
        if quantum_circuit_dict[key]['circuit'] == None:
            identity_circuit = (quantum_circuit_dict[key]['PauliWord'], quantum_circuit_dict[key]['factor'])
        else:
            circuit = quantum_circuit_dict[key]['circuit']
            raw_result = simulator.run(circuit, repetitions=num_shots)
            results[key] =raw_result.histogram(key=histogram_strings[key])


    return results,identity_circuit


def Return_as_binary(histogram_results, quantum_circuit_dict):
    """

    e.g.
    {1: Counter({3: 798, 0: 202}),
     2: Counter({2: 806, 1: 194}),
     3: Counter({1: 219, 2: 781}),
     4: Counter({2: 773, 1: 227}),
     5: Counter({1: 198, 2: 802}),
     6: Counter({3: 199, 0: 801}),
     7: Counter({1: 907, 0: 93}),
     8: Counter({1: 911, 0: 89}),
     9: Counter({0: 929, 1: 71}),
     10: Counter({0: 361, 1: 639})}

    :return:
    e.g.
    {1: {'11': 798, '00': 202},
     2: {'10': 806, '01': 194},
     3: {'01': 219, '10': 781},
     4: {'10': 773, '01': 227},
     5: {'01': 198, '10': 802},
     6: {'11': 199, '00': 801},
     7: {'1': 907, '0': 93},
     8: {'1': 911, '0': 89},
     9: {'0': 929, '1': 71},
     10: {'0': 361, '1': 639}}
    """

    results_binary={}
    for KEY in histogram_results:
        result_instance = histogram_results[KEY]

        PauliWord = quantum_circuit_dict[KEY]['PauliWord']
        PauliWord = PauliWord.split(' ')
        num_terms_to_measure = len([i for i in PauliWord if i[0] != 'I'])

        temp_result = {}
        for output_state in result_instance:
            binary_length = '{' +'0:0{}b'.format(num_terms_to_measure) + '}'

            binary_key = binary_length.format(output_state)

            temp_result[binary_key] = result_instance[output_state]

        results_binary[KEY] = {'Counter': temp_result, 'factor': quantum_circuit_dict[KEY]['factor']}
    return results_binary


def expectation_value(results_binary):
    """

    < Z >  = (num_0 - num_1) / total_num_measurements = (num_0 - num_1) / (num_0 + num_1)

    :return:
    """
    expectation_value_results ={}
    for key in results_binary:
        num_ones = 0
        num_zeros = 0
        for state in results_binary[key]['Counter']:
            individual_outcomes = [int(bit) for bit in state]
            print(individual_outcomes)
            for bit in individual_outcomes:
                if bit == 1:
                    num_ones += results_binary[key]['Counter'][state]
                elif bit == 0:
                    num_zeros += results_binary[key]['Counter'][state]
                else:
                    raise ValueError('incorrect outcome {}'.format(individual_outcomes))
        print('number of 1 s =', num_ones, 'number of 0 s =', num_zeros)
        expectation_value_results[key] = (num_zeros - num_ones) / (num_zeros + num_ones) * results_binary[key]['factor']

    return expectation_value_results



def Calc_Energy(quantum_circuit_dict, num_shots):
    histogram_results, identity_result = Simulate_Quantum_Circuit(quantum_circuit_dict, num_shots)
    results_binary = Return_as_binary(histogram_results, quantum_circuit_dict)
    exp_per_circuit = expectation_value(results_binary)

    I_result = identity_result[1]

    E= I_result
    for circuit in exp_per_circuit:
        E += exp_per_circuit[circuit]
    return E

Calc_Energy(circuits_and_constants, 1000)