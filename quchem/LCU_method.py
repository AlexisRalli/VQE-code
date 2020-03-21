# https://arxiv.org/pdf/quant-ph/0104030.pdf
# ^^^ Need to be able to prepare arbitrary state!
import numpy as np
import cirq
from functools import reduce

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
        # return cirq.CircuitDiagramInfo(
        #     wire_symbols=tuple([*['@' for _ in range(self.num_control_qubits-1)],' U = {} rad '.format(self.theta.__round__(4))]),exponent=1)
        return ' U = {} rad '.format(self.theta.__round__(4))

if __name__ == '__main__':
    theta = np.pi

    U_single_qubit = My_U_Gate(theta,0)
    op = U_single_qubit.on(cirq.LineQubit(1))
    print(cirq.Circuit(op))

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

        max_qubit = max(Tuple[0] for Tuple in self.circuit_param_dict)
        string_list = []
        for i in range(max_qubit):
            string_list.append('state prep circuit')
        return string_list

    def num_qubits(self):
        max_qubit = max(Tuple[0]+1 for Tuple in self.circuit_param_dict) # +1 due to python indexing
        return max_qubit

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

            try:
                alpha_j[(qubit_j, term)] = np.arctan(np.sqrt(sum(upper_sum) / sum(lower_sum)))
            except:
                alpha_j[(qubit_j, term)] = 0

    return alpha_j

if __name__ == '__main__':
    num_qub = 3
    # Coefficient_list = np.random.rand(2 ** num_qub) # [np.sqrt(1/(2**num_qub)) for _ in range(2**num_qub)]
    Coefficient_list = [np.sqrt(0.3), np.sqrt(0.1), np.sqrt(0.1), np.sqrt(0.1), np.sqrt(0.1), np.sqrt(0.1),
                        np.sqrt(0.1), np.sqrt(0.1)]  # [1/2,1/2,1/2,1/2]  #[0.9, 0.3, 0.3, 0.1]

    alpha_j = Get_state_prep_dict(num_qub, Coefficient_list=Coefficient_list)
    state_circ = State_Prep_Circuit(alpha_j)
    circuit = (cirq.Circuit(cirq.decompose_once((state_circ(*cirq.LineQubit.range(state_circ.num_qubits()))))))

    # circuit = cirq.Circuit(*circuit.all_operations(), *list(circuit.all_operations())[::-1])

    # MEASURE
    qubits_to_measure = (cirq.LineQubit(q_No) for q_No in range(num_qub))
    circuit.append(cirq.measure(*qubits_to_measure))
    print(circuit)

    # simulate
    simulator = cirq.Simulator()
    results = simulator.run(circuit, repetitions=100000)
    print(results.histogram(key='0,1,2')) # Need key to match number of qubits!!!
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


##

from quchem.Hamiltonian_Generator_Functions import *

### Variable Parameters
if __name__ == '__main__':
    Molecule = 'H2'  # LiH'
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.74))] # None#[('Li', (0., 0., 0.)), ('H', (0., 0., 1.45))] # None
    num_shots = 10000
    HF_occ_index = [0, 1, 2]  # [0, 1,2] # for occupied_orbitals_index_list
    #######

    ### Get Hamiltonian
    Hamilt = Hamiltonian(Molecule,
                         run_scf=1, run_mp2=1, run_cisd=1, run_ccsd=1, run_fci=1,
                         basis='sto-3g',
                         multiplicity=1,
                         geometry=geometry)  # normally None!

    Hamilt.Get_Molecular_Hamiltonian()
    SQ_CC_ops, THETA_params = Hamilt.Get_ia_and_ijab_terms(Coupled_cluser_param=True)
    # print('UCC operations: ', SQ_CC_ops)

    HF_transformations = Hamiltonian_Transforms(Hamilt.MolecularHamiltonian, SQ_CC_ops, Hamilt.molecule.n_qubits)

    QubitHam = HF_transformations.Get_Qubit_Hamiltonian_JW()
    # print('Qubit Hamiltonian: ', QubitHam)
    QubitHam_PauliStr = HF_transformations.Convert_QubitMolecularHamiltonian_To_Pauliword_Str_list(QubitHam, Hamilt.molecule.n_qubits)

    ### Graph Colouring
    from quchem.Graph import *

    List_PauliWords, HamiltonainCofactors = zip(*QubitHam_PauliStr)
    attribute_dictionary = {'Cofactors': HamiltonainCofactors}
    List_of_nodes, node_attributes_dict = Get_list_of_nodes_and_attributes(List_PauliWords,
                                                                           attribute_dictionary=attribute_dictionary)
    G = Hamiltonian_Graph(List_PauliWords, attribute_dictionary=attribute_dictionary)
    anti_commuting_sets = G.Get_Pauli_grouping('AC', Graph_colouring_strategy='largest_first', plot_graph=False)

    anti_commuting_set_stripped = Get_PauliWord_constant_tuples(anti_commuting_sets, dict_str_label='Cofactors')
    print(anti_commuting_set_stripped)


#### GUG method

### ancilla prep:

def convert(beta_k_Pk, Ps):
    convert_term = {
        'II': (1, 'I'),
        'IX': (1, 'X'),
        'IY': (1, 'Y'),
        'IZ': (1, 'Z'),

        'XI': (1, 'X'),
        'XX': (1, 'I'),
        'XY': (1j, 'Z'),
        'XZ': (-1j, 'Y'),

        'YI': (1, 'Y'),
        'YX': (-1j, 'Z'),
        'YY': (1, 'I'),
        'YZ': (1j, 'X'),

        'ZI': (1, 'Z'),
        'ZX': (1j, 'Y'),
        'ZY': (-1j, 'X'),
        'ZZ': (1, 'I')
    }

    # arXiv 1908.08067 eq (11)

    PauliWord_k = beta_k_Pk[0].split(' ')
    PauliWord_s = Ps.split(' ')

    new_PauliWord = []
    for i in range(len(PauliWord_s)):
        qubitNo = PauliWord_s[i][1::]

        if qubitNo == PauliWord_k[i][1::]:
            PauliString_s = PauliWord_s[i][0]
            PauliString_k = PauliWord_k[i][0]
            term = PauliString_s + PauliString_k
            try:
                new_PauliString = convert_term[term]
                new_PauliWord.append((new_PauliString, qubitNo))
            except:
                raise KeyError('Cannot combine: {}, as contains Non-Pauli operators'.format(term))
        else:
            raise ValueError(
                'qubit indexes do Not match. P_s index = {} and P_k index = {}'.format(qubitNo, PauliWord_k[i][1::]))

    # needed for Pauli products!
    new_constant_SIGN = np.prod([factorpaulistring[0] for factorpaulistring, qubitNo in new_PauliWord])

    seperator = ' '
    new_PauliWord = seperator.join([factorpaulistring[1] + qubitNo for factorpaulistring, qubitNo in new_PauliWord])

    return (new_PauliWord, new_constant_SIGN * beta_k_Pk[1])

from quchem.Unitary_partitioning import Get_beta_j_cofactors
# def Get_X_SET(anti_commuting_set, S_index):
#     """
#     X = i ( ∑_{k=1}^{n-1} B_{k} P_{k} ) P_{s}
#
#     X =  i( ∑_{k=1}^{n-1} B_{k} P_{ks}
#
#         where P_{ks} = P_{k} * P_{s}
#
#     note ∑_{k=1}^{n-1} B_{k}^{2} = 1
#
#     therefore have:
#     X =  gamma_l * i( ∑_{k=1}^{n-1} B_{k} P_{ks}
#
#
#     Args:
#         anti_commuting_set (list):
#         S_index (int):
#         no_qubits (int):
#     Returns:
#         LCU_dict (dict): A dictionary containing the linear combination of terms required to perform R ('R_LCU')
#                          the correction factors to make all real and positive ('LCU_correction')
#                          the angle to perform R gate ('alpha')
#                          the PauliS term ('P_s')
#      """
#
#     anti_comm_set = anti_commuting_set.copy()
#     P_S = anti_comm_set.pop(S_index)
#     H_n_1 = Get_beta_j_cofactors(anti_comm_set)  ## NOTE THIS DOESN'T CONTAIN P_S!!!
#
#     X_set={}
#     X_set['gamma_l'] = H_n_1['gamma_l']
#
#     X_set['terms'] =[]
#     for PauliWord, constant in H_n_1['PauliWords']:
#
#         new_PauliWord, new_constant = convert(( PauliWord, constant), P_S[0]) #P_{ks} term
#         X_set['terms'].append((new_PauliWord, new_constant * 1j))
#
#     X_set['P_s'] = P_S
#     X_set['H_n_1'] = H_n_1
#     return X_set
#
# def Get_R_linear_combination(anti_commuting_set, S_index, no_qubits):
#     """
#     Note
#         X = i ( ∑_{k=1}^{n-1} B_{k} P_{j} ) P_{s}
#
#         R = exp(-i ALPHA/2 X) = cos(ALPHA/2)I - i sin(ALPHA/2)X
#
#         R = cos(ALPHA/2)I + i sin(ALPHA/2) * ( ∑_{k=1}^{n-1} B_{k} P_{j} ) P_{s}
#         ^^^ - this is just a linear combiniation of Pauli terms! Can implement using LCU
#
#         H_{n} = sin(ϕ_{n-1}) * H_{n-1} +  cos(ϕ_{n-1}) * P_{s})
#         ^^^^ therefore cos(ϕ_{n-1}) = B_{s} .... SO ... ϕ_{n-1} = arc cos (B_{s})
#
#         AS:
#         R H R† = sin(ϕ_{n-1} - ALPHA) * H_{n-1} + cos(ϕ_{n-1} - APLHA) * P_{s}
#         set
#         ALPHA = ϕ_{n-1}
#
#
#         note:
#          R† (gamma_l * H) R = gamma_l * R† H R
#
#          gamma_l *H_{n-1}=  gamma_l * ∑_{k=1}^{n-1} \beta_{k} P_{k}$
#
#     Args:
#         anti_commuting_set (list):
#         S_index (int):
#         no_qubits (int):
#     Returns:
#         LCU_dict (dict): A dictionary containing the linear combination of terms required to perform R ('R_LCU')
#                          the correction factors to make all real and positive ('LCU_correction')
#                          the angle to perform R gate ('alpha')
#                          the PauliS term ('P_s')
#      """
#
#     X_set = Get_X_SET(anti_commuting_set, S_index)
#
#
#     phi_n_1 = np.arccos(X_set['P_s'][1]) # ALPHA
#     H_n = [(PauliWord, const*np.sin(phi_n_1))for PauliWord, const in X_set['H_n_1']['PauliWords']] + [(X_set['P_s'][0], np.cos(phi_n_1))]
#
#     const = np.sin(phi_n_1 / 2) * -1j
#
#     seperator = ' '
#     I_term = seperator.join(['I{}'.format(i) for i in range(no_qubits)])
#     LCU_dict = {}
#     if np.cos(phi_n_1 / 2)<0:
#         LCU_dict['R_LCU'] = {0: (I_term, np.cos(phi_n_1 / 2))}
#         LCU_dict['LCU_correction'] = {0: -1}
#     else:
#         LCU_dict['R_LCU'] = {0: (I_term, np.cos(phi_n_1 / 2))}
#         LCU_dict['LCU_correction'] = {0: 1}
#
#     # loop gets each X operator term
#     for i in range(len(X_set['terms'])):
#         P_term = X_set['terms'][i]
#         factor = P_term[1] * const
#
#         if np.iscomplex(factor) and factor.imag != 0:
#             if factor.imag < 0:
#                 LCU_dict['LCU_correction'].update({i + 1: -1j})
#                 LCU_dict['R_LCU'].update({i + 1: (P_term[0], np.abs(factor.imag))})
#             else:
#                 LCU_dict['LCU_correction'].update({i + 1: 1j})
#                 LCU_dict['R_LCU'].update({i + 1: (P_term[0], factor.imag)})
#
#         elif factor.real < 0:
#             LCU_dict['LCU_correction'].update({i + 1: -1})
#             LCU_dict['R_LCU'].update({i + 1: (P_term[0], np.abs(factor.real))})
#         else:
#             LCU_dict['LCU_correction'].update({i + 1: 1})
#             LCU_dict['R_LCU'].update({i + 1: (P_term[0], factor.real)})
#
#     if not np.isclose(sum([LCU_dict['R_LCU'][key][1]**2 for key in LCU_dict['R_LCU']]), 1):
#         raise ValueError('normalisation is WRONG: {}'.format(sum([LCU_dict['R_LCU'][key][1]**2 for key in LCU_dict['R_LCU']])))
#
#     LCU_dict['H_n'] = H_n
#     LCU_dict['alpha'] = phi_n_1
#     LCU_dict['P_s'] = X_set['P_s'][0]
#     LCU_dict['l1_norm'] = sum([LCU_dict['R_LCU'][key][1] for key in LCU_dict['R_LCU']])
#     # LCU_dict['H_n_1']= X_set['H_n_1']['PauliWords']
#
#     # LCU_dict['lambda'] = LCU_dict['l1_norm'] / sum([LCU_dict['R_LCU'][key][1]**2 + LCU_dict['R_LCU'][key][1]**2 for key in LCU_dict['R_LCU']]) # note do term**2 + term**2 (rather than just term**2)!
#
#     # LCU_dict['gamma_l'] = X_set['gamma_l']
#
#     # THIS USES gamma_l of H_n (whereas previously was using gamma_l of H_{n-1}
#
#     # H = Get_beta_j_cofactors(anti_commuting_set)
#     # gamma_l = H['gamma_l']
#
#     if X_set['P_s'][1]<0:
#         LCU_dict['gamma_l'] = (X_set['gamma_l'] * -1) # gamma_l*-1
#     else:
#         LCU_dict['gamma_l'] = X_set['gamma_l'] # gamma_l
#
#     # note overall we have:
#     # anti_commuting_set = H_n_1 * gamma_l + beta_s P_s
#
#     for PauliWord, const in X_set['H_n_1']['PauliWords']:
#         index = anti_commuting_set.index(PauliWord)
#         np.isclose(anti_commuting_set[index][1], )
#     return LCU_dict
##

# def Get_R_linear_combination(anti_commuting_set, S_index, no_qubits):
#     """
#     Note
#         X = i ( ∑_{k=1}^{n-1} B_{k} P_{j} ) P_{s}
#
#         R = exp(-i ALPHA/2 X) = cos(ALPHA/2)I - i sin(ALPHA/2)X
#
#         R = cos(ALPHA/2)I + i sin(ALPHA/2) * ( ∑_{k=1}^{n-1} B_{k} P_{j} ) P_{s}
#         ^^^ - this is just a linear combiniation of Pauli terms! Can implement using LCU
#
#         H_{n} = sin(ϕ_{n-1}) * H_{n-1} +  cos(ϕ_{n-1}) * P_{s})
#         ^^^^ therefore cos(ϕ_{n-1}) = B_{s} .... SO ... ϕ_{n-1} = arc cos (B_{s})
#
#         AS:
#         R H R† = sin(ϕ_{n-1} - ALPHA) * H_{n-1} + cos(ϕ_{n-1} - APLHA) * P_{s}
#         set
#         ALPHA = ϕ_{n-1}
#
#
#         note:
#          R† (gamma_l * H) R = gamma_l * R† H R
#
#          gamma_l *H_{n-1}=  gamma_l * ∑_{k=1}^{n-1} \beta_{k} P_{k}$
#
#     Args:
#         anti_commuting_set (list):
#         S_index (int):
#         no_qubits (int):
#     Returns:
#         LCU_dict (dict): A dictionary containing the linear combination of terms required to perform R ('R_LCU')
#                          the correction factors to make all real and positive ('LCU_correction')
#                          the angle to perform R gate ('alpha')
#                          the PauliS term ('P_s')
#      """
#     LCU_dict = {}
#
#     X_set = Get_X_SET(anti_commuting_set, S_index)
#
#     phi_n_1 = np.arccos(X_set['P_s'][1])
#     LCU_dict['alpha'] = phi_n_1
#
#     H_n = [(PauliWord,fact*np.sin(phi_n_1)) for PauliWord, fact in X_set['H_n_1']] + [(X_set['P_s'][0], np.cos(phi_n_1))]
#
#     if not np.isclose(sum( c**2for p, c in H_n), 1):
#         raise ValueError('H_n definition normalisation is WRONG')
#
#
#     seperator = ' '
#     I_term = seperator.join(['I{}'.format(i) for i in range(no_qubits)])
#     factor = np.cos(phi_n_1 / 2)
#
#     if np.iscomplex(factor) and factor.imag != 0:
#         if factor.imag < 0:
#             LCU_dict['R_LCU'] = {0: (I_term, np.abs(factor.imag))}
#             LCU_dict['LCU_correction'] = {0: -1j}
#         else:
#             LCU_dict['R_LCU'] = {0: (I_term, factor.imag)}
#             LCU_dict['LCU_correction'] = {0: 1j}
#
#     elif factor.real < 0:
#         LCU_dict['R_LCU'] = {0: (I_term, np.abs(factor.real))}
#         LCU_dict['LCU_correction'] = {0: -1}
#     else:
#         LCU_dict['R_LCU'] = {0: (I_term, factor.real)}
#         LCU_dict['LCU_correction'] = {0: 1}
#
#     # loop gets each X operator term
#     # const = np.sin(LCU_dict['alpha'] / 2) * -1j
#     const = np.sin(LCU_dict['alpha'] / 2) * 1j
#     for i in range(len(X_set['terms'])):
#         P_term = X_set['terms'][i]
#         factor = P_term[1] * const
#
#         if np.iscomplex(factor) and factor.imag != 0:
#             if factor.imag < 0:
#                 LCU_dict['LCU_correction'].update({i + 1: -1j})
#                 LCU_dict['R_LCU'].update({i + 1: (P_term[0], np.abs(factor.imag))})
#             else:
#                 LCU_dict['LCU_correction'].update({i + 1: 1j})
#                 LCU_dict['R_LCU'].update({i + 1: (P_term[0], factor.imag)})
#
#         elif factor.real < 0:
#             LCU_dict['LCU_correction'].update({i + 1: -1})
#             LCU_dict['R_LCU'].update({i + 1: (P_term[0], np.abs(factor.real))})
#         else:
#             LCU_dict['LCU_correction'].update({i + 1: 1})
#             LCU_dict['R_LCU'].update({i + 1: (P_term[0], factor.real)})
#
#     # if not np.isclose(sum([LCU_dict['R_LCU'][key][1]**2 for key in LCU_dict['R_LCU']]), 1):
#     if not np.isclose((LCU_dict['R_LCU'][0][1] * LCU_dict['LCU_correction'][0])**2 -
#                       sum((LCU_dict['R_LCU'][key][1] * LCU_dict['LCU_correction'][key])**2 for key in LCU_dict['R_LCU'] if key>0), 1):
#         raise ValueError('normalisation is WRONG: {}'.format(sum([LCU_dict['R_LCU'][key][1]**2 for key in LCU_dict['R_LCU']])))
#
#     LCU_dict['H_n'] = H_n
#     LCU_dict['P_s'] = X_set['P_s'][0]
#     LCU_dict['l1_norm'] = sum([LCU_dict['R_LCU'][key][1] for key in LCU_dict['R_LCU']])
#     LCU_dict['gamma_l'] = 1 #TODO should remove this from all future terms!
#
#     return LCU_dict
def Get_X_SET(anti_commuting_set, S_index):
    """
    X = i ( ∑_{k=1}^{n-1} B_{k} P_{k} ) P_{s}

    X =  i( ∑_{k=1}^{n-1} B_{k} P_{ks}

        where P_{ks} = P_{k} * P_{s}

    note ∑_{k=1}^{n-1} B_{k}^{2} = 1

    therefore have:
    X =  gamma_l * i( ∑_{k=1}^{n-1} B_{k} P_{ks}


    Args:
        anti_commuting_set (list):
        S_index (int):
        no_qubits (int):
    Returns:
        LCU_dict (dict): A dictionary containing the linear combination of terms required to perform R ('R_LCU')
                         the correction factors to make all real and positive ('LCU_correction')
                         the angle to perform R gate ('alpha')
                         the PauliS term ('P_s')
     """

    anti_comm_set = anti_commuting_set.copy()
    P_S = anti_comm_set.pop(S_index)
    normalised_anti_comm_set_n_1 = Get_beta_j_cofactors(anti_comm_set)  ## NOTE THIS DOESN'T CONTAIN P_S!!!

    H_n_1 = normalised_anti_comm_set_n_1['PauliWords']

    X_set={}
    X_set['terms'] =[]
    for PauliWord, constant in H_n_1:

        new_PauliWord, new_constant = convert(( PauliWord, constant), P_S[0]) #P_{ks} term
        X_set['terms'].append((new_PauliWord, new_constant * 1j))

    if not np.isclose(sum(const**2 for pauliWord, const in X_set['terms']), 1):
        raise ValueError('normalisation of X operator incorrect: {}'.format(sum(const**2 for pauliWord, const in X_set['terms'])))

    X_set['P_s'] = P_S
    X_set['H_n_1'] = H_n_1
    return X_set

def Get_R_linear_combination(anti_commuting_set, S_index, no_qubits):
    """
    Note
        X = i ( ∑_{k=1}^{n-1} B_{k} P_{j} ) P_{s}

        R = exp(-i ALPHA/2 X) = cos(ALPHA/2)I - i sin(ALPHA/2)X

        R = cos(ALPHA/2)I + i sin(ALPHA/2) * ( ∑_{k=1}^{n-1} B_{k} P_{j} ) P_{s}
        ^^^ - this is just a linear combiniation of Pauli terms! Can implement using LCU

        H_{n} = sin(ϕ_{n-1}) * H_{n-1} +  cos(ϕ_{n-1}) * P_{s})
        ^^^^ therefore cos(ϕ_{n-1}) = B_{s} .... SO ... ϕ_{n-1} = arc cos (B_{s})

        AS:
        R H R† = sin(ϕ_{n-1} - ALPHA) * H_{n-1} + cos(ϕ_{n-1} - APLHA) * P_{s}
        set
        ALPHA = ϕ_{n-1}


        note:
         R† (gamma_l * H) R = gamma_l * R† H R

         gamma_l *H_{n-1}=  gamma_l * ∑_{k=1}^{n-1} \beta_{k} P_{k}$

    Args:
        anti_commuting_set (list):
        S_index (int):
        no_qubits (int):
    Returns:
        LCU_dict (dict): A dictionary containing the linear combination of terms required to perform R ('R_LCU')
                         the correction factors to make all real and positive ('LCU_correction')
                         the angle to perform R gate ('alpha')
                         the PauliS term ('P_s')
     """
    LCU_dict = {}
    X_set = Get_X_SET(anti_commuting_set, S_index)

    phi_n_1 = np.arccos(X_set['P_s'][1])
    LCU_dict['alpha'] = phi_n_1

    H_n = [(PauliWord,fact*np.sin(phi_n_1)) for PauliWord, fact in X_set['H_n_1']] + [(X_set['P_s'][0], np.cos(phi_n_1))]

    if not np.isclose(sum(c**2for p, c in H_n), 1):
        raise ValueError('H_n definition normalisation is WRONG')

    LCU_dict['H_n'] = H_n

    seperator = ' '
    I_term = seperator.join(['I{}'.format(i) for i in range(no_qubits)])
    I_P_word= (I_term, np.cos(LCU_dict['alpha'] / 2))

    R_Op = [I_P_word]

    # loop gets each X operator term
    const = np.sin(LCU_dict['alpha'] / 2) * -1j
    for P_term, factor in X_set['terms']:
        new_factor = factor * const

        R_Op.append((P_term, new_factor))

    if not np.isclose(sum(abs(const)**2for PauliWord, const in R_Op), 1):
        raise ValueError('R_operator definition normalisation is WRONG: {}'.format(sum(abs(const)**2for PauliWord, const in R_Op)))

    LCU_dict['R_Op'] = R_Op
    # need all constants of R_op to be positive and real for LCU
    # hence next absorb phases into operator and have correct dict

    LCU_dict['R_LCU']={}
    LCU_dict['LCU_correction']={}
    for index, (PauliWord, factor) in enumerate(R_Op):

        if np.iscomplex(factor) and factor.imag != 0:
            if factor.imag < 0:
                LCU_dict['LCU_correction'].update({index: -1j})
                LCU_dict['R_LCU'].update({index: (PauliWord, np.abs(factor.imag))})
            else:
                LCU_dict['LCU_correction'].update({index: 1j})
                LCU_dict['R_LCU'].update({index: (PauliWord, factor.imag)})

        elif factor.real < 0:
            LCU_dict['LCU_correction'].update({index: -1})
            LCU_dict['R_LCU'].update({index: (PauliWord, np.abs(factor.real))})
        else:
            LCU_dict['LCU_correction'].update({index: 1})
            LCU_dict['R_LCU'].update({index: (PauliWord, factor.real)})

    if not np.isclose(sum(abs(LCU_dict['R_LCU'][key][1])**2 for key in LCU_dict['R_LCU']), 1):
        raise ValueError('R_operator definition normalisation is WRONG')

    # LCU_dict['H_n'] = H_n
    LCU_dict['P_s'] = X_set['P_s'][0]
    LCU_dict['l1_norm'] = sum([LCU_dict['R_LCU'][key][1] for key in LCU_dict['R_LCU']])
    # LCU_dict['gamma_l'] = 1  # TODO should remove this from all future terms!

    #this is the omega term in jupyter notebook!
    LCU_dict['gamma_l'] = np.cos(LCU_dict['alpha'] / 2) / X_set['P_s'][1]

    # if X_set['P_s'][1]<0:
    #     LCU_dict['gamma_l'] = -1
    # else:
    #     LCU_dict['gamma_l'] = 1


    # analytical approach to (R H R† = Pn) becoming ==> R H = Pn R
    LCU_dict['RH_n'] = [(PauliWord, fact * np.sin(3*LCU_dict['alpha']/2)) for PauliWord, fact in X_set['H_n_1']] + [(X_set['P_s'][0], np.cos(3*LCU_dict['alpha']/2))]

    return LCU_dict

# ITERATIVE approach to (R H R† = Pn) becoming ==> R H = Pn R
def convert_new(beta_k_Pk, beta_j_Pj):
    convert_term = {
        'II': (1, 'I'),
        'IX': (1, 'X'),
        'IY': (1, 'Y'),
        'IZ': (1, 'Z'),

        'XI': (1, 'X'),
        'XX': (1, 'I'),
        'XY': (1j, 'Z'),
        'XZ': (-1j, 'Y'),

        'YI': (1, 'Y'),
        'YX': (-1j, 'Z'),
        'YY': (1, 'I'),
        'YZ': (1j, 'X'),

        'ZI': (1, 'Z'),
        'ZX': (1j, 'Y'),
        'ZY': (-1j, 'X'),
        'ZZ': (1, 'I')
    }

    # arXiv 1908.08067 eq (11)

    PauliWord_k = beta_k_Pk[0].split(' ')
    PauliWord_s = beta_j_Pj[0].split(' ')

    new_PauliWord = []
    for i in range(len(PauliWord_s)):
        qubitNo = PauliWord_s[i][1::]

        if qubitNo == PauliWord_k[i][1::]:
            PauliString_s = PauliWord_s[i][0]
            PauliString_k = PauliWord_k[i][0]
            term = PauliString_s + PauliString_k
            try:
                new_PauliString = convert_term[term]
                new_PauliWord.append((new_PauliString, qubitNo))
            except:
                raise KeyError('Cannot combine: {}, as contains Non-Pauli operators'.format(term))
        else:
            raise ValueError(
                'qubit indexes do Not match. P_s index = {} and P_k index = {}'.format(qubitNo, PauliWord_k[i][1::]))

    # needed for Pauli products!
    new_constant_SIGN = np.prod([factorpaulistring[0] for factorpaulistring, qubitNo in new_PauliWord])

    seperator = ' '
    new_PauliWord = seperator.join([factorpaulistring[1] + qubitNo for factorpaulistring, qubitNo in new_PauliWord])

    return (new_PauliWord, new_constant_SIGN * beta_k_Pk[1]* beta_j_Pj[1])
def Get_R_times_Hn_terms(LCU_dict):
    """
    Takes in LCU dict and looks at the following

     R H R† = P_{s}

    R H = P_{s} R

    Therefore only need to implement R ------ P_{s} ------ M
    on quantum computer

    Function used to find what new object we are measuring (aka R H)...
    currently makes sure that R H = P_{s} R is also true!

    args:
        threshold (optional, float): gives threshold of terms to ignore... e.g. the term
                                    (0.00003+0j) [Y0 X1 X2 Y3]] would be ignored if threshold = 1e-2

    returns:
        R H (dict):

    """

    R_terms = LCU_dict['R_Op']
    H_n = LCU_dict['H_n']

    # new_terms = {}
    # for PauliWord, const in H_n:
    #     for key in R_terms:
    #
    #         Pauli_R_term = (R_terms[key][0], (R_terms[key][1] * LCU_dict['LCU_correction'][key]))
    #
    #         new_term = convert_new(Pauli_R_term, (PauliWord, const))
    #
    #         if new_term[0] in new_terms.keys():
    #             new_terms[new_term[0]] -= new_term[1] #TODO check this sign!
    #         else:
    #             new_terms[new_term[0]] = new_term[1]
    #
    # P_S = (LCU_dict['P_s'], 1)
    #
    # new_terms2 = {}
    # for key in R_terms:
    #
    #     Pauli_R_term = (R_terms[key][0], (R_terms[key][1] * LCU_dict['LCU_correction'][key]))
    #
    #     new_term = convert_new(P_S, Pauli_R_term)
    #
    #     if new_term[0] in new_terms2.keys():
    #         new_terms2[new_term[0]] += new_term[1]  #TODO check this sign!
    #     else:
    #         new_terms2[new_term[0]] = new_term[1]
    #
    # print(new_terms, 'VS', new_terms2)
    # return [(PauliWord, const) for PauliWord, const in new_terms.items()]

    R_H_n = {}
    for PwordConst_Hn in H_n:
        for PwordConst_R in R_terms:

            new_PauilWord, new_const = convert_new(PwordConst_R, PwordConst_Hn)

            if new_PauilWord in R_H_n.keys():
                R_H_n[new_PauilWord] += new_const
            else:
                R_H_n[new_PauilWord] = new_const

    R_H_n = [(PauliWord_key, R_H_n[PauliWord_key])for PauliWord_key in R_H_n if not np.isclose(R_H_n[PauliWord_key] ,0)]

    return R_H_n


def Get_ancilla_amplitudes(LCU_dict):
    """
    Takes in an anti_commuting set and returns l1 norm, number of ancilla qubits and amplitudes required
    on ancilla bits.

    Note that anti_commuting_set = H_s =  ∑_{j=1}^{d} a_{j} U_{j}

    where ||a_vec||_{1} = ∑_{j=1}^{d} |a_{j}| # this is the l1 norm

    need all a_{j}>=0 (complex and negative sign can be absorbed by U_{j}

    Args:
        LCU_dict (dict): A dictionary containing R_LCU_terms (key = 'R_LCU') -- > from Get_R_linear_comb funciton
    Returns:
        l1 norm (float): L1 norm
        number_ancilla_qubits (int): number of ancilla qubits
        G_ancilla_amplitudes (list): List of ancilla amplitudes

    state = |000> + |001> + |010> + |011> + |100> + |101 > + |110 > + |111>
    state  = |0> +   |1> +   |2> +   |3> +   |4> +   |5 > +   |6 > +   |7>

   example_input =
            [
                 ('I0 I1 I2 I3 I4 I5 I6 I7 I8 Z9 I10 I11', (-0.2167542932500046+0j)),
                 ('I0 I1 I2 I3 I4 I5 X6 Y7 Y8 X9 I10 I11', (0.004217284878422757+0j)),
                 ('Y0 Y1 I2 I3 I4 I5 I6 I7 X8 X9 I10 I11', (-0.002472706153881526+0j)),
                 ('Y0 Z1 Z2 Y3 I4 I5 I6 I7 X8 X9 I10 I11', (0.002077887498395704+0j)),
                 ('Y0 Z1 Z2 Z3 Z4 Y5 I6 I7 X8 X9 I10 I11', (0.0025623897800114877+0j)),
                 ('Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11', (0.000908468961622956+0j))
             ]


    """

    # NOTE each coefficient must be positive... sign should be absorbed by operator!
    l1_norm = LCU_dict['l1_norm']
    number_ancilla_qubits = int(np.ceil(np.log2(len(LCU_dict['R_LCU']))))  # note round up with np.ceil

    l1_normalised_amp = [LCU_dict['R_LCU'][key][1] / l1_norm for key in LCU_dict['R_LCU']]

    G_ancilla_amplitudes = [np.sqrt(amp) for amp in l1_normalised_amp]

    if len(G_ancilla_amplitudes) < 2 ** number_ancilla_qubits:
        G_ancilla_amplitudes += [0 for _ in range(int(2 ** number_ancilla_qubits - len(G_ancilla_amplitudes)))]

    # # check <<- slows down code though!
    # for key in LCU['R_LCU']:
    #     w_l = LCU['R_LCU'][key][1]
    #     if not np.isclose(w_l, LCU['l1_norm']*G_ancilla_amplitudes[key]**2):
    #         print(LCU['l1_norm']*G_ancilla_amplitudes[key]**2)
    #         raise ValueError('Wrong ancilla amplitudes or l1 norm')

    return l1_norm, number_ancilla_qubits, G_ancilla_amplitudes


if __name__ == '__main__':
    test_set = anti_commuting_set_stripped[7]
    S_index=0
    LCU = Get_R_linear_combination(test_set, S_index, Hamilt.molecule.n_qubits)

## Build state:

if __name__ == '__main__':
    l1_norm_val, number_ancilla_qubits, G_ancilla_amplitudes = Get_ancilla_amplitudes(LCU)
    ancilla_line_qubits = cirq.LineQubit.range(Hamilt.molecule.n_qubits,
                                               Hamilt.molecule.n_qubits + number_ancilla_qubits)

    alpha_j = Get_state_prep_dict(number_ancilla_qubits, Coefficient_list=G_ancilla_amplitudes)
    state_circ = State_Prep_Circuit(alpha_j)
    circuit = (cirq.Circuit(cirq.decompose_once((state_circ(*ancilla_line_qubits)))))
    # MEASURE
    circuit.append(cirq.measure(*ancilla_line_qubits))
    print(circuit)

    # looking at state:
    simulator = cirq.Simulator()
    state_circ = State_Prep_Circuit(alpha_j)
    circuit = (cirq.Circuit(cirq.decompose_once((state_circ(*ancilla_line_qubits)))))
    qubits_to_measure = (cirq.LineQubit(q_No) for q_No in
                         range(Hamilt.molecule.n_qubits, Hamilt.molecule.n_qubits + number_ancilla_qubits))
    result = simulator.simulate(circuit, qubit_order=qubits_to_measure)
    print(np.around(result.final_state, 3))
    print('expected state amplitudes:', G_ancilla_amplitudes)


### LCU_GUG method


class Change_modified_PauliStr_gate_to_Z_basis(cirq.SingleQubitGate):
    """
    Take in a string of PauliWord ('X', 'Y', 'Z', 'I')
    e.g. X and correction factor e.g. 1i to give overall modified operator: iX

    The function finds eigenvalue of operator and THEN gives corresponding operator to change to Z basis for measurement!

    Args:
        LCU_PauliWord_and_cofactor (tuple): Tuple of PauliWord (str) and constant (complex) ... (PauliWord, constant)
        LCU_correction_value (complex):

    Returns
        A cirq circuit object to be used by cirq.Circuit

    """

    def __init__(self, Pauli_str, Correction_value):

        self.Pauli_str = Pauli_str
        self.Correction_value = Correction_value

    def _unitary_(self):

        from scipy.linalg import eig

        if self.Pauli_str == 'Z':
            P_mod = cirq.Z._unitary_() * self.Correction_value

        elif self.Pauli_str == 'Y':
            P_mod = cirq.Y._unitary_() * self.Correction_value

        elif self.Pauli_str == 'X':
            P_mod = cirq.X._unitary_() * self.Correction_value

        elif self.Pauli_str == 'I':
            P_mod = cirq.I._unitary_() * self.Correction_value

        else:
            raise TypeError('not a Pauli operation')
        val, P_mod_eig = eig(P_mod)

        x1 = P_mod_eig[0,0]
        x2 = P_mod_eig[1,0]
        x3 = P_mod_eig[0,1]
        x4 = P_mod_eig[1,1]

        a= (-x4*x3)/((-x3*x1*x4)+(x3**2*x2))
        b= (x3)/((-x1*x4)+(x3*x2))
        c= (-x2*x1)/((-x1*x2*x3)+(x1**2*x4))
        d= (x1)/((-x2*x3)+(x1*x4))
        U = np.array([[a,b], [c,d]])

        tol = 1e-8
        U.real[abs(U.real) < tol] = 0.0
        U.imag[abs(U.imag) < tol] = 0.0
        return U

    def num_qubits(self):
        return 1

    def _circuit_diagram_info_(self, args):
        return 'change to Z basis for modified PauliMod : {} times {}'.format(self.Pauli_str, self.Correction_value)

class Change_modified_PauliStr_gate_to_Z_basis(cirq.SingleQubitGate):
    """
    Take in a string of PauliWord ('X', 'Y', 'Z', 'I')
    e.g. X and correction factor e.g. 1i to give overall modified operator: iX

    The function finds eigenvalue of operator and THEN gives corresponding operator to change to Z basis for measurement!

    Args:
        LCU_PauliWord_and_cofactor (tuple): Tuple of PauliWord (str) and constant (complex) ... (PauliWord, constant)
        LCU_correction_value (complex):

    Returns
        A cirq circuit object to be used by cirq.Circuit

    """

    def __init__(self, Pauli_str, Correction_value):

        self.Pauli_str = Pauli_str
        self.Correction_value = Correction_value

    def _unitary_(self):

        from scipy.linalg import eig

        if self.Pauli_str == 'Z':
            P_mod = cirq.Z._unitary_() * self.Correction_value

        elif self.Pauli_str == 'Y':
            P_mod = cirq.Y._unitary_() * self.Correction_value

        elif self.Pauli_str == 'X':
            P_mod = cirq.X._unitary_() * self.Correction_value

        elif self.Pauli_str == 'I':
            P_mod = cirq.I._unitary_() * self.Correction_value

        else:
            raise TypeError('not a Pauli operation')




    def num_qubits(self):
        return 1

    def _circuit_diagram_info_(self, args):
        return 'change to Z basis for modified PauliMod : {} times {}'.format(self.Pauli_str, self.Correction_value)


if __name__ == '__main__':
    Pauli_str = 'Z'
    Correction_value = -1

    test_gate = Change_modified_PauliStr_gate_to_Z_basis(Pauli_str, Correction_value)
    circ = test_gate.on(cirq.LineQubit(1))
    print(cirq.Circuit(circ))



class Perform_Modified_PauliWord(cirq.Gate):
    """
    Class to generate cirq circuit as gate that performs a modified PauliWord

    Args:
        LCU_PauliWord_and_cofactor (tuple): Tuple of PauliWord (str) and constant (complex) ... (PauliWord, constant)
        LCU_correction_value (complex):

    Returns
        A cirq circuit object to be used by cirq.Circuit

    e.g.
        test_case = ('Y0 Z1 X2 I3', 0.00070859248123462)
        correction_val = (-0 - 1j)
        dag = False

        P_circ_mod = Perform_Modified_PauliWord(test_case, correction_val, dag)
        print(cirq.Circuit(
        cirq.decompose_once((P_circ_mod(*cirq.LineQubit.range(P_circ_mod.num_qubits()))))))
        >>
                0: ───change to Z basis for modified PauliMod : Y times -1j───
                1: ───change to Z basis for modified PauliMod : Z times -1j───
                2: ───change to Z basis for modified PauliMod : X times -1j───
                3: ───change to Z basis for modified PauliMod : I times -1j───

    """

    def __init__(self, LCU_PauliWord_and_cofactor, LCU_correction_value):

        self.PauliWord_and_cofactor = LCU_PauliWord_and_cofactor
        self.correction_value = LCU_correction_value

    def _decompose_(self, qubits):

        PauliWord = self.PauliWord_and_cofactor[0].split(' ')

        for PauliString in PauliWord:
            qubitOp = PauliString[0]
            qubitNo = int(PauliString[1::])

            yield Change_modified_PauliStr_gate_to_Z_basis(qubitOp, self.correction_value).on(qubits[qubitNo])

    def _circuit_diagram_info_(self, args):
        string_list = []
        PauliWord = self.PauliWord_and_cofactor[0].split(' ')
        for _ in range(len(PauliWord)):
            string_list.append('change to Z basis for Pauliword_modified_for_LCU ')
        return string_list

    def num_qubits(self):
        PauliWord = self.PauliWord_and_cofactor[0].split(' ')
        return len(PauliWord)


if __name__ == '__main__':
    test_case = ('Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Y11', 0.00070859248123462)
    correction_val = (-0 - 1j)

    P_circ_mod = Perform_Modified_PauliWord(test_case, correction_val)

    print(cirq.Circuit((P_circ_mod(*cirq.LineQubit.range(P_circ_mod.num_qubits())))))
    print(
        cirq.Circuit(
            cirq.decompose_once((P_circ_mod(*cirq.LineQubit.range(P_circ_mod.num_qubits()))))))


class LCU_R_gate(cirq.Gate):
    """
    Function to build cirq Circuit that performs controlled modified pauligate for LCU method


    Args:
        circuit_param_dict (dict): A Dictionary of Tuples (qubit, control_val(int)) value is angle

    Returns
        A cirq circuit object to be used by cirq.Circuit.from_ops to generate arbitrary state

    """

    def __init__(self, LCU_dict, dagger, No_control_qubits, No_system_qubits):

        self.LCU_dict = LCU_dict
        self.dagger = dagger
        self.No_control_qubits = No_control_qubits
        self.No_system_qubits = No_system_qubits


    def _decompose_(self, qubits):

        if self.dagger is False:
            state_index = 0
            for control_state in self.LCU_dict['R_LCU']:
                control_str = Get_state_as_str(self.No_control_qubits, state_index)
                # using state index rather than control_state value
                # due to how LCU_term is built! (aka missing keys cause problems!)

                control_values = [int(bit) for bit in control_str]

                correction_val = self.LCU_dict['LCU_correction'][control_state]
                mod_P = self.LCU_dict['R_LCU'][control_state]
                mod_p_word_gate = Perform_Modified_PauliWord(mod_P, correction_val, self.dagger)

                qubit_list = cirq.LineQubit.range(self.No_system_qubits, self.No_system_qubits + self.No_control_qubits) \
                             + cirq.LineQubit.range(self.No_system_qubits)  # note control qubits first!

                state_index += 1

                yield mod_p_word_gate.controlled(num_controls=self.No_control_qubits, control_values=control_values).on(
                    *qubit_list)  # *qubit_list
        else:
            state_index = len(self.LCU_dict['R_LCU'])-1
            for control_state in list(self.LCU_dict['R_LCU'].keys())[::-1]:
                control_str = Get_state_as_str(self.No_control_qubits, state_index)
                # using state index rather than control_state value
                # due to how LCU_term is built! (aka missing keys cause problems!)

                control_values = [int(bit) for bit in control_str]

                correction_val = self.LCU_dict['LCU_correction'][control_state]
                mod_P = self.LCU_dict['R_LCU'][control_state]
                mod_p_word_gate = Perform_Modified_PauliWord(mod_P, correction_val, self.dagger)

                qubit_list = cirq.LineQubit.range(self.No_system_qubits, self.No_system_qubits + self.No_control_qubits) \
                             + cirq.LineQubit.range(self.No_system_qubits)  # note control qubits first!

                state_index -= 1

                yield mod_p_word_gate.controlled(num_controls=self.No_control_qubits, control_values=control_values).on(
                    *qubit_list)  # *qubit_list

    def _circuit_diagram_info_(self, args):

        string_list = []
        if self.dagger:
            for _ in range(self.No_system_qubits):
                string_list.append('Pauli_Mod_Cirq_LCU_DAGGER')
        else:
            for _ in range(self.No_system_qubits):
                string_list.append('Pauli_Mod_Cirq_LCU')

        for _ in range(self.No_control_qubits):
            string_list.append('control_LCU')

        return string_list

    def num_qubits(self):
        return self.No_control_qubits + self.No_system_qubits

    def Get_each_l1_combination(self):
        # l1_combined_terms=[]
        # for control_state in self.LCU_dict['R_LCU']:
        #     l1_combined_terms.append(self.LCU_dict['l1_norm'])
        # return l1_combined_terms
        return [self.LCU_dict['l1_norm']]


if __name__ == '__main__':
    dag = False
    GATE = LCU_R_gate(LCU, dag, number_ancilla_qubits, Hamilt.molecule.n_qubits)

    print(cirq.Circuit((GATE(*cirq.LineQubit.range(GATE.num_qubits())))))
    print(
        cirq.Circuit(
            cirq.decompose_once((GATE(*cirq.LineQubit.range(GATE.num_qubits()))))))


class Measure_system_and_ancilla(cirq.Gate):
    """
    Class to generate cirq circuit that measures PauliWord in Z BASIS AND ancilla line!!!!

    e.g.: PauliWord_and_cofactor = ('X0 Y1 Z2 I3 Y4', -0.28527408634774526j)
          n_ancilla_qubits = 2

        gives :
                0: ───M───
                      │
                1: ───M───
                      │
                2: ───M───
                      │
                4: ───M───
                      │
                5: ───M───
                      │
                6: ───M───

    Args:
        PauliWord_and_cofactor (tuple): Tuple of PauliWord (str) and constant (complex) ... (PauliWord, constant)
        n_ancilla_qubits (int): Number of ancilla qubits

    Returns
        A cirq circuit object to be used by cirq.Circuit

    """
    def __init__(self, PauliWord_and_cofactor, n_ancilla_qubits):

        self.PauliWord_and_cofactor = PauliWord_and_cofactor
        self.n_ancilla_qubits = n_ancilla_qubits

    def _decompose_(self, qubits):

        q_No_measure = []  # list of qubits to measure!
        for PauliString in self.PauliWord_and_cofactor[0].split(' '):
            qubitOp = PauliString[0]
            qubitNo = int(PauliString[1::])

            if qubitOp in ['X', 'Y', 'Z']:
                q_No_measure.append(qubitNo)
            elif qubitOp == 'I':
                continue
            else:
                raise ValueError("Qubit Operation: {} is NOT a Pauli operation".format(qubitOp))

        for i in np.arange(len(self.PauliWord_and_cofactor[0].split(' ')),
                           len(self.PauliWord_and_cofactor[0].split(' ')) + self.n_ancilla_qubits, 1):
            q_No_measure.append(i)

        qubits_to_measure = (qubits[q_No] for q_No in q_No_measure)
        if q_No_measure != []:
            yield cirq.measure(*qubits_to_measure)
        else:
            return None

    def _circuit_diagram_info_(self, args):
        string_list = []
        PauliWord = self.PauliWord_and_cofactor[0].split(' ')
        for i in range(len(PauliWord)+self.n_ancilla_qubits):
            string_list.append(' Measuring_PauliWord_and_ancilla ')
        return string_list

    def num_qubits(self):
        PauliWord = self.PauliWord_and_cofactor[0].split(' ')
        return len(PauliWord) + self.n_ancilla_qubits

# def add_Pauli_Z_to_Ancilla(anti_commuting_sets, n_ancilla, n_system):
#     """
#     Function to take in anti_commuting_sets and add pauli Z terms to ancilla qubits.
#
#
#     Args:
#         anti_commuting_sets (dict): A Dictionary of Tuples (PauliWord, const)
#         n_ancilla (int): number of ancilla qubits
#         n_system (int): number of system qubits
#
#     Returns
#         anti_commuting_sets_with_ancilla (dict): A Dictionary of Tuples (PauliWord_with_ancilla_Z, const)
#
#     """
#     ancilla_list = ['Z{}'.format(i) for i in np.arange(n_system, n_system + n_ancilla)]
#     seperator = ' '
#     ancilla_str = seperator.join(ancilla_list)
#
#     anti_commuting_set_with_ancilla={}
#     for key in anti_commuting_sets:
#         for PauliWord, const in anti_commuting_sets[key]:
#             NEW_PauliWord = PauliWord + ' ' + ancilla_str
#             anti_commuting_set_with_ancilla[key] = (NEW_PauliWord, const)
#     return anti_commuting_set_with_ancilla


if __name__ == '__main__':
    P_test = ('X0 Y1 Z2 I3 Y4', -0.28527408634774526j)
    n_ancilla=3
    Measure_obj = Measure_system_and_ancilla(P_test,n_ancilla)

    print(cirq.Circuit((Measure_obj(*cirq.LineQubit.range(Measure_obj.num_qubits())))))
    print(
        cirq.Circuit(
            cirq.decompose_once((Measure_obj(*cirq.LineQubit.range(Measure_obj.num_qubits()))))))


from quchem.quantum_circuit_functions import *

def Complete_LCU_circuit(anti_commuting_set, No_system_qubits, S_index):
    """
     Descript

    Args:
        gg

    Returns:
        full_circuit (cirq.circuits.circuit.Circuit):

    """

    if len(anti_commuting_set) < 2:
        # cannot perform LCU on set with only 1 term in it!
        return None
    else:
        # LCU_DICT = Get_R_linear_comb(anti_commuting_set, S_index, No_system_qubits)
        LCU_DICT = Get_R_linear_combination(anti_commuting_set, S_index, No_system_qubits)

        l1_norm_val, number_ancilla_qubits, G_ancilla_amplitudes = Get_ancilla_amplitudes(LCU_DICT)

        alpha_j = Get_state_prep_dict(number_ancilla_qubits, Coefficient_list=G_ancilla_amplitudes)
        state_circ = State_Prep_Circuit(alpha_j)
        ancilla_line_qubits = cirq.LineQubit.range(No_system_qubits, No_system_qubits + number_ancilla_qubits)

        G_prep_circuit = (cirq.Circuit(cirq.decompose_once((state_circ(*ancilla_line_qubits)))))

        G_prep_circuit_UNDO = cirq.Circuit(list(G_prep_circuit.all_operations())[::-1]) # reverses!

        # G_prep_circuit.append(cirq.measure(*ancilla_line_qubits)) # add measurement gates to ancilla line

        R_gate_obj = LCU_R_gate(LCU_DICT, False, number_ancilla_qubits, No_system_qubits)
        R_gate = cirq.Circuit(
            cirq.decompose_once((R_gate_obj(*cirq.LineQubit.range(R_gate_obj.num_qubits())))))

        Pauli_S = anti_commuting_set[S_index]

        Pauli_S_circ_obj = Perform_PauliWord(Pauli_S)
        Pauli_S_circ = cirq.Circuit(
            cirq.decompose_once((Pauli_S_circ_obj(
                *cirq.LineQubit.range(Pauli_S_circ_obj.num_qubits())))))

        change_basis_PS = Change_Basis_to_Measure_PauliWord(Pauli_S)
        change_basis_PS_circuit = cirq.Circuit(
            cirq.decompose_once((change_basis_PS(*cirq.LineQubit.range(change_basis_PS.num_qubits())))))

        # R_dagger_gate_obj = LCU_R_gate(LCU_DICT, True, number_ancilla_qubits, No_system_qubits)
        # R_dagger_gate = cirq.Circuit(
        #     cirq.decompose_once((R_dagger_gate_obj(*cirq.LineQubit.range(R_dagger_gate_obj.num_qubits())))))

        measure_gate_obj = Measure_system_and_ancilla(Pauli_S,number_ancilla_qubits)

        measure_circ = cirq.Circuit(
            cirq.decompose_once((measure_gate_obj(*cirq.LineQubit.range(measure_gate_obj.num_qubits())))))

        full_circuit = cirq.Circuit(
            [
                G_prep_circuit.all_operations(),
                *R_gate.all_operations(),
                Pauli_S_circ.all_operations(),
                change_basis_PS_circuit.all_operations(),
                # R_dagger_gate.all_operations(),
                G_prep_circuit_UNDO.all_operations(),
                measure_circ.all_operations()
            ]
        )

        # if Pauli_S[1]<0:
        #     gamma_l = LCU_DICT['gamma_l'] *-1
        # else:
        #     gamma_l = LCU_DICT['gamma_l']
        l1_R_circuits = R_gate_obj.Get_each_l1_combination()
        # l1_R_dagger_circuits = R_dagger_gate_obj.Get_each_l1_combination()

        ####

        # # LCU_DICT = Get_R_linear_comb(anti_commuting_set, S_index, No_system_qubits)
        # LCU_DICT = Get_R_linear_combination(anti_commuting_set, S_index, No_system_qubits)
        #
        # l1_norm_val, number_ancilla_qubits, G_ancilla_amplitudes = Get_ancilla_amplitudes(LCU_DICT)
        #
        # alpha_j = Get_state_prep_dict(number_ancilla_qubits, Coefficient_list=G_ancilla_amplitudes)
        # state_circ = State_Prep_Circuit(alpha_j)
        # ancilla_line_qubits = cirq.LineQubit.range(No_system_qubits, No_system_qubits + number_ancilla_qubits)
        #
        # G_prep_circuit = (cirq.Circuit(cirq.decompose_once((state_circ(*ancilla_line_qubits)))))
        #
        # G_prep_circuit_UNDO = cirq.Circuit(list(G_prep_circuit.all_operations())[::-1]) # reverses!
        #
        # # G_prep_circuit.append(cirq.measure(*ancilla_line_qubits)) # add measurement gates to ancilla line
        #
        # R_dagger_gate_obj = LCU_R_gate(LCU_DICT, True, number_ancilla_qubits, No_system_qubits)
        # R_dagger_gate = cirq.Circuit(
        #     cirq.decompose_once((R_dagger_gate_obj(*cirq.LineQubit.range(R_dagger_gate_obj.num_qubits())))))
        #
        # Pauli_S = anti_commuting_set[S_index]
        #
        # Pauli_S_circ_obj = Perform_PauliWord(Pauli_S)
        # Pauli_S_circ = cirq.Circuit(
        #     cirq.decompose_once((Pauli_S_circ_obj(
        #         *cirq.LineQubit.range(Pauli_S_circ_obj.num_qubits())))))
        #
        # R_gate_obj = LCU_R_gate(LCU_DICT, False, number_ancilla_qubits, No_system_qubits)
        # R_gate = cirq.Circuit(
        #     cirq.decompose_once((R_gate_obj(*cirq.LineQubit.range(R_gate_obj.num_qubits())))))
        #
        # measure_gate_obj = Measure_system_and_ancilla(Pauli_S,number_ancilla_qubits)
        #
        # measure_circ = cirq.Circuit(
        #     cirq.decompose_once((measure_gate_obj(*cirq.LineQubit.range(measure_gate_obj.num_qubits())))))
        #
        # full_circuit = cirq.Circuit(
        #     [
        #         G_prep_circuit.all_operations(),
        #         *R_gate.all_operations(),
        #         Pauli_S_circ.all_operations(),
        #         R_dagger_gate.all_operations(),
        #         G_prep_circuit_UNDO.all_operations(),
        #         measure_circ.all_operations()
        #     ]
        # )
        #
        # # if Pauli_S[1]<0:
        # #     gamma_l = LCU_DICT['gamma_l'] *-1
        # # else:
        # #     gamma_l = LCU_DICT['gamma_l']
        # l1_R_circuits = R_gate_obj.Get_each_l1_combination()
        # l1_R_dagger_circuits = R_dagger_gate_obj.Get_each_l1_combination()
        #
        # return full_circuit, LCU_DICT['gamma_l'], l1_R_circuits + l1_R_dagger_circuits, number_ancilla_qubits
        return full_circuit, LCU_DICT['gamma_l'], l1_R_circuits, number_ancilla_qubits # full_circuit, LCU_DICT['gamma_l'], l1_R_circuits + l1_R_dagger_circuits, number_ancilla_qubits


if __name__ == '__main__':
    test_set = anti_commuting_set_stripped[7]
    S_index = 0
    circuit, gamma, l1_norm, number_ancilla_qubits = Complete_LCU_circuit(test_set, Hamilt.molecule.n_qubits, S_index)
    print(circuit)




# LCU GUG method



def ALCU_dict(Full_Ansatz_Q_Circuit, anti_commuting_sets, S_dict,
                                                     n_system):
    """
     Function that appends Ansatz Quantum Circuit to Pauli perform and measure circuit instance.

    Args:
        Full_Ansatz_Q_Circuit (cirq.circuits.circuit.Circuit): Full cirq Ansatz Q circuit
        anti_commuting_sets (dict): dict of list of tuples containing PauliWord to perform and constant -> (PauliWord, constant)
        S_dict (dict): dict of PauliWordS indices
        n_system (int): number of system qubits

    Returns:
        dic_holder (dict): Returns a dictionary of each quantum circuit, with cofactor, PauliWord and cirq Q Circuit

    """

    I_Measure = ['I{}'.format(i) for i in range(n_system)]
    seperator = ' '
    PauliWord_I_only = seperator.join(I_Measure)


    dic_holder = {}

    for key in anti_commuting_sets:
        temp_d = {}
        PauliString_and_Constant = anti_commuting_sets[key]
        S_index = S_dict[key]

        if PauliString_and_Constant[0][0] == PauliWord_I_only:
            temp_d['circuit'] = None
            temp_d['gamma_l'] = PauliString_and_Constant[0][1]
            temp_d['PauliWord'] = PauliString_and_Constant[0][0]
            temp_d['LCU'] = False

        elif len(PauliString_and_Constant) == 1:
            Q_circuit_obj = Perform_PauliWord_and_Measure(PauliString_and_Constant[0])#QubitHam_PauliStr[key])
            Q_circuit = cirq.Circuit(
                    cirq.decompose_once(
                        (Q_circuit_obj(*cirq.LineQubit.range(Q_circuit_obj.num_qubits())))))

            full_circuit = cirq.Circuit(
                [
                    Full_Ansatz_Q_Circuit,
                    Q_circuit
                ]
            )

            temp_d['circuit'] = full_circuit
            temp_d['gamma_l'] = PauliString_and_Constant[0][1]
            temp_d['PauliWord'] = PauliString_and_Constant[0][0]
            temp_d['LCU'] = False
        else:
            # ALCU_circuit, gamma_l, l1_norm = Complete_LCU_circuit(PauliString_and_Constant, n_system, S_index)
            ALCU_circuit, gamma_l, all_l1_norms, number_ancilla_qubits = Complete_LCU_circuit(PauliString_and_Constant, n_system, S_index)

            full_circuit = cirq.Circuit(
                [
                    Full_Ansatz_Q_Circuit,
                    ALCU_circuit
                ]
            )

            ancilla_list = ['Z{}'.format(int(i)) for i in np.arange(n_system, n_system + number_ancilla_qubits)]
            seperator = ' '
            ancilla_str = seperator.join(ancilla_list)

            temp_d['circuit'] = full_circuit
            temp_d['PauliWord'] = PauliString_and_Constant[S_index][0] + ' ' + ancilla_str
            temp_d['LCU'] = True
            temp_d['gamma_l'] = gamma_l #* PauliString_and_Constant[S_index][1].real #TODO check this here!!!
            # temp_d['l1_norm'] = l1_norm
            temp_d['all_l1_norms'] = all_l1_norms
            temp_d['n_ancilla'] = number_ancilla_qubits

        dic_holder[key] = temp_d

    return dic_holder



from quchem.Simulating_Quantum_Circuit import *
from functools import reduce
class ALCU_Simulation_Quantum_Circuit_DictRAW():
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


    def SimulateQC_RAW(self, Quantum_Circuit,key):
        simulator = cirq.Simulator()
        l1_multiplied = reduce((lambda x, y: x * y), self.circuits_factor_PauliWord_dict[key]['all_l1_norms'])
        p_success = (1/l1_multiplied)**2
        raw_result = simulator.run(Quantum_Circuit, repetitions=self.num_shots*int(np.ceil(1/p_success.real)))
        return raw_result

    def Get_binary_results_dict(self):
        if self.hist_key_dict is None:
            self.Get_Histkey_dict()

        binary_results_dict = {}
        for key in self.circuits_factor_PauliWord_dict:
            #print(key)
            if self.hist_key_dict[key] != '':
                #checks for non identity ciruict
                if self.circuits_factor_PauliWord_dict[key]['LCU'] is False:
                    counter_result = Simulate_Quantum_Circuit(self.circuits_factor_PauliWord_dict[key]['circuit'],
                                                                self.num_shots, self.hist_key_dict[key])
                    binary_results_dict[key] = Return_as_binary(counter_result, self.circuits_factor_PauliWord_dict[key]['PauliWord'])
                else:
                    LCU_result_dict = {}
                    correct_ancilla_state = np.zeros([self.circuits_factor_PauliWord_dict[key]['n_ancilla']])
                    n_success_shots=0
                    while n_success_shots != self.num_shots:
                        raw_result = self.SimulateQC_RAW(self.circuits_factor_PauliWord_dict[key]['circuit'], key)
                        M_results = raw_result.measurements[self.hist_key_dict[key]]

                        for result in M_results:
                            if np.array_equal(result[:(-1-self.circuits_factor_PauliWord_dict[key]['n_ancilla']):-1][::-1], correct_ancilla_state): # Checks if all zero ancilla measured!
                                seperator = ''
                                state_key_binary = seperator.join(map(str, result[0:-self.circuits_factor_PauliWord_dict[key]['n_ancilla']])) #Gets rid of ancilla part!!!
                                if state_key_binary not in LCU_result_dict.keys():
                                    LCU_result_dict[state_key_binary] = 1
                                else:
                                    LCU_result_dict[state_key_binary] += 1
                                n_success_shots += 1
                            # else:
                            #     print('fail!!')

                            if n_success_shots == self.num_shots:
                                break
                    binary_results_dict[key] = LCU_result_dict
            else:
                self.Identity_result_dict[key]= (self.circuits_factor_PauliWord_dict[key]['PauliWord'], self.circuits_factor_PauliWord_dict[key]['gamma_l'])

        self.binary_results_dict = binary_results_dict

    def Get_expectation_value_via_parity(self):
        if self.binary_results_dict is None:
            self.Get_binary_results_dict()

        expect_results_dict = {}
        for key in self.circuits_factor_PauliWord_dict:

            if key in self.Identity_result_dict.keys():
                expect_results_dict[key] = 1


            # elif self.circuits_factor_PauliWord_dict[key]['LCU']:
            #     # remove ancilla results with slice!
            #     # ancilla_removed = {state[0:-1*self.n_ancilla]: counts for state, counts in self.binary_results_dict[key].items()}
            #     expect_results_dict[key] = expectation_value_by_parity(ancilla_removed)
            # else:
            #     expect_results_dict[key] = expectation_value_by_parity(self.binary_results_dict[key])
            else:
                expect_results_dict[key] = expectation_value_by_parity(self.binary_results_dict[key])

        self.expect_results_dict = expect_results_dict

    def Calc_energy_via_parity(self):
        if self.expect_results_dict is None:
            self.Get_expectation_value_via_parity()

        Energy_list =[]
        for key in self.circuits_factor_PauliWord_dict:
            if self.circuits_factor_PauliWord_dict[key]['LCU'] is False:
                exp_val = self.expect_results_dict[key]
                factor = self.circuits_factor_PauliWord_dict[key]['gamma_l']
                Energy_list.append((exp_val*factor))
            else:
                exp_val = self.expect_results_dict[key]
                factor = self.circuits_factor_PauliWord_dict[key]['gamma_l']
                # Energy_list.append((exp_val * factor*self.circuits_factor_PauliWord_dict[key]['l1_norm']))

                # l1_multiplied = reduce((lambda x, y: x * y),  self.circuits_factor_PauliWord_dict[key]['all_l1_norms']) # self.circuits_factor_PauliWord_dict[key]['all_l1_norms']#
                # Energy_list.append((exp_val * factor *l1_multiplied)) #TODO not sure if use l1_multiplied AS have R and R_DAGGER (not sure wether dagger has an effect!)
                # Energy_list.append((exp_val *1)/factor) #TODO currently IGNORING L1_multiplied!
                Energy_list.append(exp_val)

        self.Energy_list = Energy_list
        self.Energy = sum(Energy_list)

        return self.Energy # * self.l1_norm #TODO mistake HERE OR



import time
def time_funct(func):
    def wrapper(*args, **kwargs):
        t_start = time.time()
        result = func(*args, **kwargs)
        t_end = time.time()

        total_t = t_end-t_start

        print("run time: {}".format(total_t))

        return result#,total_t
    return wrapper


def notes():
    """
    Note
        X = i ( ∑_{k=1}^{n-1} B_{k} P_{j} ) P_{s}

        R = exp(-i ALPHA/2 X) = cos(ALPHA/2)I - i sin(ALPHA/2)X

        R = cos(ALPHA/2)I + i sin(ALPHA/2) * ( ∑_{k=1}^{n-1} B_{k} P_{j} ) P_{s}
        ^^^ - this is just a linear combiniation of Pauli terms! Can implement using LCU

        H_{n} = sin(ϕ_{n-1}) * H_{n-1} +  cos(ϕ_{n-1}) * P_{s})
        ^^^^ therefore cos(ϕ_{n-1}) = B_{s} .... SO ... ϕ_{n-1} = arc cos (B_{s})

        AS:
        R† H R = sin(ϕ_{n-1} - ALPHA) * H_{n-1} + cos(ϕ_{n-1} - APLHA) * P_{s}

        set
        ALPHA = ϕ_{n-1}
        gives:

        Hn= R† Pn R

    """

    pass





# if __name__ == '__main__':
#     ####
#     # TODO checking with linear algebra!!! THIS SHOULD MAKE METHOD CLEAR :)
#
#     ### get H_n matrix
#     set_index = 9
#     test_set = anti_commuting_set_stripped[set_index].copy()
#     S_index=0
#     P_S = test_set.pop(S_index)
#     H_n_1 = Get_beta_j_cofactors(test_set) ## NOTE THIS DOESN'T CONTAIN P_S!!!
#
#     phi_n_1 = np.arccos(P_S[1])
#     H_n=[]
#     for Pword, const in H_n_1['PauliWords']:
#         H_n.append((Pword, (np.sin(phi_n_1) *const)))
#     H_n.append((P_S[0], np.cos(phi_n_1)))
#     print(sum([const**2 for Pword, const in H_n]))
#
#
#
#
#     from scipy.sparse import csr_matrix, kron
#     from functools import reduce
#
#     I = csr_matrix(np.eye(2))
#     X = csr_matrix(np.array([
#                             [0,1],
#                             [1,0]]))
#     Y = csr_matrix(np.array([
#                             [0,-1j],
#                             [1j,0]]))
#     Z = csr_matrix(np.array([
#                             [1,0],
#                             [0,-1]]))
#
#     PauliDICT={
#         'I': I,
#         'X': X,
#         'Z': Z,
#         'Y': Y
#     }
#
#     n_qubits = Hamilt.molecule.n_qubits
#     H_n_MATRIX=csr_matrix(np.zeros([2**n_qubits, 2**n_qubits]))
#     for term in H_n:
#         PauliWord = term[0]
#         const = term[1]
#         P_strings = [PauliDICT[sig[0]] for sig in PauliWord.split(' ')]
#         mat = reduce(kron, P_strings)
#         H_n_MATRIX += mat*const
#
#
#     # Get R_matrix
#     qq = Get_R_linear_combination(anti_commuting_set_stripped[set_index], S_index, Hamilt.molecule.n_qubits)
#     R_matrix=csr_matrix(np.zeros([2**n_qubits, 2**n_qubits]))
#     for key in qq['R_LCU']:
#         PauliWord = qq['R_LCU'][key][0]
#         const = qq['R_LCU'][key][1]
#         correction = qq['LCU_correction'][key]
#         P_strings = [PauliDICT[sig[0]] for sig in PauliWord.split(' ')]
#         mat = reduce(kron, P_strings)
#         R_matrix += mat * const * correction
#
#
#     # Alternate method (CHECK FOR EQUIVALENCE)
#     X_SET = Get_X_SET(anti_commuting_set_stripped[set_index], S_index)
#     from scipy.sparse.linalg import expm
#     X_matrix=csr_matrix(np.zeros([2**n_qubits, 2**n_qubits]))
#     for term in X_SET['terms']:
#         PauliWord = term[0]
#         const = term[1]
#
#         P_strings = [PauliDICT[sig[0]] for sig in PauliWord.split(' ')]
#         mat = reduce(kron, P_strings)
#         X_matrix += mat*const
#
#     R_MATRIX = expm(-1j * qq['alpha']/2 * X_matrix)
#
#     # check if the same!!
#     print(np.allclose(R_matrix.todense(), R_MATRIX.todense()))
#
#
#     ######
#     # NOW check  R * Hn * R† =  Ps
#
#     first = R_matrix.dot(H_n_MATRIX)
#     second = first.dot(R_matrix.transpose().conj())
#     # reduce(np.matmul, [R_matrix.todense(), H_n_MATRIX.todense(), R_matrix.transpose().conj().todense()])
#
#     ### PS
#     Ps_string = [PauliDICT[sig[0]] for sig in qq['P_s'].split(' ')]
#     P_s_MATRIX = reduce(kron, Ps_string) * np.cos(phi_n_1- qq['alpha'])
#
#     print(np.allclose(second.todense(), P_s_MATRIX.todense()))
#
#     ##
#
#
#
#     ll = R_matrix.todense() == R_MATRIX.todense()
#     np.where(ll==False)
#     print(R_matrix.todense()[1,14])
#     print(R_MATRIX.todense()[1,14])
#
#


# if __name__ == '__main__':
#     ####
#     from scipy.sparse import csr_matrix, kron
#     from functools import reduce
#
#     zero = csr_matrix(np.array([[1],[0]]))
#     one = csr_matrix(np.array([[0],[1]]))
#
#     Q_states={
#         '0': zero,
#         '1': one
#     }
#
#     PauliDICT = {
#         'I': I,
#         'X': X,
#         'Z': Z,
#         'Y': Y
#     }
#
#     def Arb_state_Gen_Matrix(state_amplitudes, n_ancilla):
#         MATRIX = csr_matrix(np.zeros([2 ** n_ancilla, 2 ** n_ancilla]))
#
#         state_amp = csr_matrix(np.array(state_amplitudes).reshape([len(state_amplitudes),1]))
#
#         MATRIX = scipy.sparse.hstack((state_amp, MATRIX[:,1:]))
#         return MATRIX
#
#     if __name__ == '__main__':
#         zero_all_ancilla = [Q_states['0'] for _ in range(2)]
#         zero_ancilla_state = reduce(kron, zero_all_ancilla)
#
#         z = Arb_state_Gen_Matrix([0.25, 0.25, 0.25, 0.25], 2)
#         print(z.dot(zero_ancilla_state).todense())
#
#     def Get_ancilla_prep_and_system_I(state_amplitudes, n_ancilla, n_system):
#         ancilla_matrix = Arb_state_Gen_Matrix(state_amplitudes, n_ancilla)
#
#         N_system = csr_matrix(np.eye(2 ** n_system))
#
#         return kron(N_system, ancilla_matrix)
#
#     def Get_R_LCU_matrix(Pauli_SET, n_ancilla):
#         # # U = ∑_{i} P_{i} ⊗ |i>  <i|
#         # #           system.....ancilla
#         R_LCU_matrices=[]
#         for i in range(len(Pauli_SET)):
#             ancilla_state_str = Get_state_as_str(n_ancilla, i)
#             ancilla_state = [Q_states[bit] for bit in ancilla_state_str]
#             ancilla_matrix = reduce(kron, ancilla_state)
#
#             ancilla_part = ancilla_matrix.dot(ancilla_matrix.transpose().conj()) # |i>  <i|
#
#             P_term = Pauli_SET[i]
#             system_P = [PauliDICT[Pauli[0]] for Pauli in P_term.split(' ')]
#             system_matrix = reduce(kron, system_P)  #  P_{i} ⊗ |i>  <i|
#
#             R_LCU_matrices.append(kron(system_matrix, ancilla_part))
#
#         return R_LCU_matrices
#
#     if __name__ == '__main__':
#         # ['I0 I1 I2 Z3', 'X0 Y1 Y2 X3']
#         R_LCU_Mat_list = Get_R_LCU_matrix(['I0 Z1', 'X0 Y1'], 2)
#
#     test_set = anti_commuting_set_stripped[7]
#     S_index=0
#     LCU_DICT = Get_R_linear_combination(test_set, S_index, Hamilt.molecule.n_qubits)
#     l1_norm_val, number_ancilla_qubits, G_ancilla_amplitudes = Get_ancilla_amplitudes(LCU_DICT)
#
#
#     state_prep = Arb_state_Gen_Matrix(G_ancilla_amplitudes, number_ancilla_qubits)
#     I_system = csr_matrix(np.eye(2 ** Hamilt.molecule.n_qubits))
#     state_prep_system_and_ancilla = kron(I_system, state_prep)
#
#
#     R_LCU_matrix_list = Get_R_LCU_matrix([tup[0] for tup in test_set], number_ancilla_qubits)
#
#     Ps_string = [PauliDICT[sig[0]] for sig in LCU_DICT['P_s'].split(' ')]
#     P_s_MATRIX = reduce(kron, Ps_string) * np.cos(phi_n_1 - LCU_DICT['alpha'])
#     # need to tensor this with idenity on ancilla line!!! ^^^^
#     I_ancilla = csr_matrix(np.eye(2 ** number_ancilla_qubits))
#     P_s_MATRIX_and_ancilla = kron(P_s_MATRIX, I_ancilla)
#
#
#
#     HF_transformations = Hamiltonian_Transforms(Hamilt.MolecularHamiltonian, SQ_CC_ops, Hamilt.molecule.n_qubits)
#     UCC_JW_excitation_matrix_list = HF_transformations.Get_Jordan_Wigner_CC_Matrices()
#
#     THETA_parameters=[1,2,3]
#     from scipy.sparse.linalg import expm
#     UCC_matrices_and_ANCILLA=[]
#     I_ancilla = csr_matrix(np.eye(2 ** number_ancilla_qubits))
#     for k in reversed(range(0, len(THETA_parameters))):
#         SYSTEM_AND_ANCILLA = kron(UCC_JW_excitation_matrix_list[k], I_ancilla)
#         UCC_matrices_and_ANCILLA.append(expm((THETA_parameters[k] * SYSTEM_AND_ANCILLA))) #TODO may need to do theta/2 ect...
#
#
#
#
#     # all_Zero = [Q_states['0'] for _ in range(number_ancilla_qubits + Hamilt.molecule.n_qubits)]
#     # state = reduce(kron, all_Zero)
#     from quchem.Ansatz_Generator_Functions import HF_state_generator
#     HF_state_obj = HF_state_generator(Hamilt.molecule.n_electrons, Hamilt.molecule.n_qubits)
#     HF_state = HF_state_obj.Get_JW_HF_vector()
#     HF_state = HF_state.reshape([HF_state.shape[0], 1])
#
#     zero_ancilla = [Q_states['0'] for _ in range(number_ancilla_qubits)]
#     zero_ancilla_state = reduce(kron, zero_ancilla)
#
#     state = kron(HF_state, zero_ancilla_state)
#
#     OP_list = [state_prep_system_and_ancilla, *UCC_matrices_and_ANCILLA,  *R_LCU_matrix_list, P_s_MATRIX_and_ancilla, *R_LCU_matrix_list[::-1], state_prep_system_and_ancilla]
#
#     for OP in OP_list:
#         state = OP.dot(state)
#
#     if (state.todense() != 0).any():
#         print(state.todense())