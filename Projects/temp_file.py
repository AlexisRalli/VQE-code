import numpy as np
normalised_anticommuting_set_DICT = {
                                            'PauliWords': [   ('Z0 I1 I2 I3', (0.8918294488900189+0j)),
                                                              ('Y0 X1 X2 Y3', (0.3198751585326103+0j)),
                                                              ('X0 I1 I2 I3', (0.3198751585326103+0j))],
                                            'gamma_l': (0.1538026463340925+0j)
                                            }

def thetasFromOplist(normalisedOplist):
    betas = [x[1] for x in normalisedOplist]
    squaredBetas = [x**2 for x in betas]

    runningTotal = squaredBetas[-1]
    squaredBetaSums = [runningTotal]
    for i in range(1,len(normalisedOplist)-1):
        runningTotal += squaredBetas[i-1]
        squaredBetaSums.append(runningTotal)

    l2Betas = [x**(1./2.) for x in squaredBetaSums]
    l2Betas[0] = betas[-1]
    thetas = [np.arctan(betas[i]/l2Betas[i]) for i in range(len(l2Betas))]
    if betas[-1].real < 0.:
        thetas[0] = thetas[0] + np.pi
    return thetas

def convert_X_sk(X_sk):
    """

    Converts P_s, P_k tuple into the corresponding X_sk term (PauliWord, correction_factor).
    Where X_sk = i P_s P_k [note that beta cofactors omitted in definition. When multiplying the PauliWords,
    they gain different cofactors, YX = -1i Z . This effect is taken into account by this function and the overall
    effect is returned as the correction factor.

    Args:
        X_sk (tuple): A tuple of (Pauliword_s, Pauliword_k) where each is a tuple of (PauliWord, constant)

    Returns:
        tuple: i* (P_s P_k) as a (Pauliword, constant). Note that constant here is NOT cofactor from Hamiltonian
               but in fact the correction term from tensor all the Paulis. e.g. YX = -1i Z.

    .. code-block:: python
       :emphasize-lines: 7

       from quchem.Unitary_partitioning import *
       X_sk = (
              ('Z0 I1 I2 I3', (0.8918294488900189+0j)), # P_s
              ('Y0 X1 X2 Y3', (0.3198751585326103+0j))  # P_k
            )

       convert_X_sk(X_sk)
       >> ('X0 X1 X2 Y3', (1+0j))

    """
    convert_term ={
        'II': (1,'I'),
        'IX': (1,'X'),
        'IY': (1,'Y'),
        'IZ': (1,'Z'),

        'XI': (1,'X'),
        'XX': (1,'I'),
        'XY': (1j,'Z'),
        'XZ': (-1j,'Y'),

        'YI': (1,'Y'),
        'YX': (-1j,'Z'),
        'YY': (1,'I'),
        'YZ': (1j,'X'),

        'ZI': (1,'Z'),
        'ZX': (1j,'Y'),
        'ZY': (-1j,'X'),
        'ZZ': (1,'I')
    }

    # arXiv 1908.08067 eq (11)
    new_constant = 1j

    PauliWord_s = X_sk[0][0].split(' ')
    PauliWord_k = X_sk[1][0].split(' ')

    new_PauliWord = []
    for i in range(len(PauliWord_s)):
        qubitNo = PauliWord_s[i][1::]

        if qubitNo == PauliWord_k[i][1::]:
            PauliString_s =  PauliWord_s[i][0]
            PauliString_k = PauliWord_k[i][0]

            term = PauliString_s + PauliString_k

            try:
                new_PauliString = convert_term[term]
                new_PauliWord.append((new_PauliString, qubitNo))
            except:
                raise KeyError('Cannot combine: {}, as contains Non-Pauli operators'.format(term))
        else:
            raise ValueError('qubit indexes do Not match. P_s index = {} and P_k index = {}'.format(qubitNo, PauliWord_k[i][1::]))

    # needed for Pauli products!
    new_constant_SIGN = np.prod([factorpaulistring[0] for factorpaulistring, qubitNo in new_PauliWord])

    seperator = ' '
    new_PauliWord = seperator.join([factorpaulistring[1] + qubitNo for factorpaulistring, qubitNo in new_PauliWord])

    return (new_PauliWord, new_constant_SIGN*new_constant)
def Get_X_sk_operators(normalised_anticommuting_set_DICT, S=0): # TODO write function to select 'best' S term!
    """

    Function takes in a normalised_anti_commuting_set, which is a list of PauliWord tuples (PauliWord, constant),
    and returns each R_sk operator according to eq (11) in ArXiv:1908.08067.

    Args:
        normalised_anticommuting_set_DICT (list): A list of Pauliwords, where each entry is a tuple of (PauliWord, constant)
        S (optional, int) = index for PauliWord_S term. #TODO

    Returns:
        dict: A dictionary of 'PauliWord_S' yields (PauliWord, correction_factor_due_matrix_multiplication), t
        he normalisation correction value 'gamma_l' (complex) and each 'X_sk_theta_sk'... which is a list of
        dictionaries that are defined with 'X_sk' = (PauliWord, correction_factor_due_matrix_multiplication) and
        'theta_sk' is rotational angle in radians. NOTE: each element of X_sk_theta_sk dict is a list of sub
        dictionaries each associated to one sk term.

    .. code-block:: python
       :emphasize-lines: 9

       from quchem.Unitary_partitioning import *
       normalised_anticommuting_set_DICT = {
                                            'PauliWords': [   ('Z0 I1 I2 I3', (0.8918294488900189+0j)),
                                                              ('Y0 X1 X2 Y3', (0.3198751585326103+0j)),
                                                              ('X0 I1 I2 I3', (0.3198751585326103+0j))   ],
                                            'gamma_l': (0.1538026463340925+0j)
                                            }

       Get_X_sk_operators(normalised_anticommuting_set_DICT, S=0)
        >> {
             'X_sk_theta_sk': [   {'X_sk': ('X0 X1 X2 Y3', (1+0j)), 'theta_sk': (0.34438034648829496+0j)},
                                    {'X_sk': ('Y0 I1 I2 I3', (-1+0j)), 'theta_sk': (0.325597719954341+0j)}
                                ],
             'PauliWord_S': ('Z0 I1 I2 I3', (1+0j)),
             'gamma_l': (0.1538026463340925+0j)
           }

    """

    anti_commuting_set = normalised_anticommuting_set_DICT['PauliWords']

    if len(anti_commuting_set) > 1:

        k_indexes = [index for index in range(len(anti_commuting_set)) if
                   index != S]

        Op_list = []
        beta_S = anti_commuting_set[S][1]

        for k in k_indexes:
            X_sk_op =(anti_commuting_set[S], anti_commuting_set[k])

            beta_K = anti_commuting_set[k][1]

            tan_theta_sk = beta_K / beta_S
            theta_sk = np.arctan(tan_theta_sk)
            Op_list.append({'X_sk': convert_X_sk(X_sk_op), 'theta_sk': theta_sk})

            beta_S = beta_K*np.sin(theta_sk) + beta_S*np.cos(theta_sk)



        return {'X_sk_theta_sk': Op_list, 'PauliWord_S': (anti_commuting_set[S][0], beta_S), 'gamma_l': normalised_anticommuting_set_DICT['gamma_l']}


### our codes agree here
print('ANDREW CODE: ', thetasFromOplist(normalised_anticommuting_set_DICT['PauliWords']))
My_result = Get_X_sk_operators(normalised_anticommuting_set_DICT, S=2)
print('ALEXIS CODE: ', [d['theta_sk'] for d in My_result['X_sk_theta_sk']])


print('####\n BUT \n#####')

normalised_anticommuting_set_DICT = {
                                            'PauliWords': [('Z0 I1 I2 I3', (-0.8918294488900189+0j)),       # added minus
                                                              ('Y0 X1 X2 Y3', (0.3198751585326103+0j)),
                                                              ('X0 I1 I2 I3', (-0.3198751585326103+0j))],   # added minus
                                            'gamma_l': (0.1538026463340925+0j)
                                            }

# they do NOT agree here
print('ANDREW CODE: ', thetasFromOplist(normalised_anticommuting_set_DICT['PauliWords']))
My_result = Get_X_sk_operators(normalised_anticommuting_set_DICT, S=2)
print('ALEXIS CODE: ', [d['theta_sk'] for d in My_result['X_sk_theta_sk']])

# here first term differs by pi according to thetas[0] = thetas[0] + np.pi correction in your code
# these are equivalent:
# tan(theta + pi)  == tan(theta)

# BUT second angle term differs by a sign... I believe this is due to sign in front of first B_k term (k=0)


from quchem.quantum_circuit_functions import *
theta=1

P_S_circuit = full_exponentiated_PauliWord_circuit(My_result['PauliWord_S'], theta)
P_circuit = cirq.Circuit.from_ops(cirq.decompose_once(P_S_circuit(*cirq.LineQubit.range(P_S_circuit.num_qubits()))))


from quchem.Unitary_partitioning import *
MY_RS_circuits_dagger = Get_R_S_operators(My_result, dagger=True)
MY_RS_circuits = Get_R_S_operators(My_result, dagger=False)

my_RS_dagger = [cirq.Circuit.from_ops(cirq.decompose_once(circ['q_circuit'](*cirq.LineQubit.range(circ['q_circuit'].num_qubits())))) for circ in MY_RS_circuits_dagger]
my_RS = [cirq.Circuit.from_ops(cirq.decompose_once(circ['q_circuit'](*cirq.LineQubit.range(circ['q_circuit'].num_qubits())))) for circ in MY_RS_circuits]

Measure = Change_Basis_and_Measure_PauliWord(My_result['PauliWord_S'])
Measure_circuit = cirq.Circuit.from_ops(cirq.decompose_once(Measure(*cirq.LineQubit.range(Measure.num_qubits()))))

andrew_result = {'X_sk_theta_sk': [{'X_sk': ('Y0 I1 I2 I3', (1+0j)),
   'theta_sk': (4.368008633896395+0j)},
  {'X_sk': ('Z0 X1 X2 Y3', (-1+0j)), 'theta_sk': (0.325597719954341+0j)}],
 'PauliWord_S': ('X0 I1 I2 I3', (-1+0j)),
 'gamma_l': (0.1538026463340925+0j)}

Andrew_RS_circuits_dagger = Get_R_S_operators(andrew_result, dagger=True)
Andrew_RS_circuits = Get_R_S_operators(andrew_result, dagger=False)

Andrew_RS_dagger = [cirq.Circuit.from_ops(cirq.decompose_once(circ['q_circuit'](*cirq.LineQubit.range(circ['q_circuit'].num_qubits())))) for circ in Andrew_RS_circuits_dagger]
Andrew_RS = [cirq.Circuit.from_ops(cirq.decompose_once(circ['q_circuit'](*cirq.LineQubit.range(circ['q_circuit'].num_qubits())))) for circ in Andrew_RS_circuits]



from quchem.Ansatz_Generator_Functions import *
HF_initial_state= HF_state_generator(2, 4)
HF_initial_state_circuit = State_Prep(HF_initial_state)
HF_circuit = cirq.Circuit.from_ops(cirq.decompose_once(HF_initial_state_circuit(*cirq.LineQubit.range(HF_initial_state_circuit.num_qubits()))))




My_circuit = cirq.Circuit.from_ops(
                        [
                            *list(HF_circuit.all_operations()),
                            *[list(q_circ.all_operations())for q_circ in my_RS_dagger],
                            *list(P_circuit.all_operations()),
                            *[list(q_circ.all_operations())for q_circ in my_RS],
                            *list(Measure_circuit.all_operations())
                        ])



###
Andrew_circuit = cirq.Circuit.from_ops(
                        [
                            *list(HF_circuit.all_operations()),
                            *[list(q_circ.all_operations())for q_circ in Andrew_RS_dagger],
                            *list(P_circuit.all_operations()),
                            *[list(q_circ.all_operations())for q_circ in Andrew_RS],
                            *list(Measure_circuit.all_operations())
                        ])


from quchem.Simulating_Quantum_Circuit import *
My_Sim = Simulate_Single_Circuit(My_result['PauliWord_S'][0], My_circuit, 10000)
print(My_Sim.Get_expectation_value_via_parity())

Andrew_Sim = Simulate_Single_Circuit(My_result['PauliWord_S'][0], Andrew_circuit, 10000)
print(Andrew_Sim.Get_expectation_value_via_parity())



Andrew_circ_dict = {
                    0: {'circuit': Andrew_circuit,
                        'gamma_l': My_result['gamma_l'],
                      'PauliWord': My_result['PauliWord_S'][0]}
                }

x = Simulation_Quantum_Circuit_Dict(Andrew_circ_dict, 100000)
x.Calc_energy_via_parity()
print(x.Energy)


My_circ_dict = {
                    0: {'circuit': My_circuit,
                        'gamma_l': My_result['gamma_l'],
                      'PauliWord': My_result['PauliWord_S'][0]}
                }

y = Simulation_Quantum_Circuit_Dict(My_circ_dict, 100000)
y.Calc_energy_via_parity()
print(y.Energy)