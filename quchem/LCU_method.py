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
    Molecule = 'LiH'  # LiH'
    geometry = None
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
    QubitHam_PauliStr = HF_transformations.Convert_QubitMolecularHamiltonian_To_Pauliword_Str_list(QubitHam)

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


def Get_R_linear_comb(anti_commuting_set, S_index, no_qubits):
    """
    Note
        X = i ( ∑_{k=1}^{n-1} B_{k} P_{j} ) P_{s}

        R = exp(-i THETA/2 X) = cos(a/2)I - i sin(a/2)X

        R = cos(THETA/2)I + i sin(THETA/2) * ( ∑_{k=1}^{n-1} B_{k} P_{j} ) P_{s}
        ^^^ - this is just a linear combiniation of Pauli terms! Can implement using LCU

        THETA = arc cos(B_{s}) <-- coefficient of P_s

    Args:
        anti_commuting_set (list):
        S_index (int):
        no_qubits (int):
    Returns:
        LCU_terms (dict): dict of tuples (PauliWord, positive constant (no complex or neg))
        LCU_correction_dict (dict): correction to make constant positive and real

     """
    # TODO need to absorb complex phase into operator!!!!!!!
    Pauli_S = anti_commuting_set[S_index][0]

    seperator = ' '
    I_term = seperator.join(['I{}'.format(i) for i in range(no_qubits)])

    THETA = np.arccos(anti_commuting_set[S_index][1])

    const = np.sin(THETA / 2)

    LCU_terms = {0: (I_term, np.cos(THETA / 2))}
    LCU_correction_dict = {0: 1}
    for i in range(len(anti_commuting_set)):
        if i != S_index:
            P_term = convert(anti_commuting_set[i], Pauli_S)

            factor = P_term[1] * const

            if np.iscomplex(factor) and factor.imag != 0:
                if factor.imag < 0:
                    LCU_correction_dict[i + 1] = -1j
                    LCU_terms[i + 1] = (P_term[0], np.abs(factor.imag))
                else:
                    LCU_correction_dict[i + 1] = 1j
                    LCU_terms[i + 1] = (P_term[0], factor.imag)

            elif factor.real < 0:
                LCU_correction_dict[i + 1] = -1
                LCU_terms[i + 1] = (P_term[0], np.abs(factor.real))
            else:
                LCU_correction_dict[i + 1] = 1
                LCU_terms[i + 1] = (P_term[0], factor.real)
    return LCU_terms, LCU_correction_dict


# def Get_R_linear_comb(anti_commuting_set, S_index, no_qubits):
#     """
#         Note
#             X = i ( ∑_{k=1}^{n-1} B_{k} P_{j} ) P_{s}
#
#             R = exp(-i THETA/2 X) = cos(a/2)I - i sin(a/2)X
#
#             R = cos(THETA/2)I + i sin(THETA/2) * ( ∑_{k=1}^{n-1} B_{k} P_{j} ) P_{s}
#             ^^^ - this is just a linear combiniation of Pauli terms! Can implement using LCU
#
#             THETA = arc cos(B_{s}) <-- coefficient of P_s
#      """
#     # TODO need to absorb complex phase into operator!!!!!!!
#     Pauli_S = anti_commuting_set[S_index][0]
#
#     seperator = ' '
#     I_term = seperator.join(['I{}'.format(i) for i in range(no_qubits)])
#
#     THETA = np.arccos(anti_commuting_set[S_index][1])
#
#     const=np.sin(THETA/2)
#
#     terms=[ (I_term, np.cos(THETA/2)) ]
#
#     for i in range(len(anti_commuting_set)):
#         if i!=S_index:
#             P_term = convert(anti_commuting_set[i], Pauli_S)
#
#             factor = P_term[1]*const
#             terms.append((P_term[0], factor)) # TODO want to only have positive real part!
#     return terms
#
# test_set = anti_commuting_set_stripped[6]
# tt = Get_R_linear_comb(test_set, 0, Hamilt.molecule.n_qubits)

def Get_ancilla_amplitudes(LCU_terms):
    """
    Takes in an anti_commuting set and returns l1 norm, number of ancilla qubits and amplitudes required
    on ancilla bits.

    Note that anti_commuting_set = H_s =  ∑_{j=1}^{d} a_{j} U_{j}

    where ||a_vec||_{1} = ∑_{j=1}^{d} |a_{j}| # this is the l1 norm

    need all a_{j}>=0 (complex and negative sign can be absorbed by U_{j}

    Args:
        LCU_terms (dict): dictionary with values as tuples (PauliWord, constant)
    Returns:
        l1 norm (float), number of ancilla qubits (int) and list of ancilla amplitudes

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
    l1_norm = sum([LCU_terms[key][1] for key in LCU_terms])
    number_ancilla_qubits = int(np.ceil(np.log2(len(LCU_terms))))  # note round up with np.ceil

    l1_normalised_amp = [LCU_terms[key][1] / l1_norm for key in LCU_terms]
    G_ancilla_amplitudes = [np.sqrt(amp) for amp in l1_normalised_amp]

    if len(G_ancilla_amplitudes) < 2 ** number_ancilla_qubits:
        G_ancilla_amplitudes += [0 for _ in range(int(2 ** number_ancilla_qubits - len(G_ancilla_amplitudes)))]

    return l1_norm, number_ancilla_qubits, G_ancilla_amplitudes


if __name__ == '__main__':
    test_set = anti_commuting_set_stripped[6]
    tt, correction_list = Get_R_linear_comb(test_set, 0, Hamilt.molecule.n_qubits)
    l1_norm_val, number_ancilla_qubits, G_ancilla_amplitudes = Get_ancilla_amplitudes(tt)

## Build state:

if __name__ == '__main__':
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

class PauliModified_gate(cirq.SingleQubitGate):
    """
    Take in a string of PauliWord ('X', 'Y', 'Z', 'I')

    NOTE that for iPsPk term = X_sk = ('X0 X1 X2 Y3', -(1+0j))
                                                       ^^^^ this is the correction factor!

    Info on matrix definition found at: https://arxiv.org/pdf/1001.3855.pdf


    Args:
        theta_sk (float): angle to rotate by in radians.
        dagger (bool): Whether to have dagger or non dagger quantum gate
        correction_factor (optional, complex): Correction value from X_sk operator.
                                               e.g. if X_sk = ('X0 X1 X2 Y3', (-1+0j)) then it would be -1.
                                              (due to X_sk = i*P_s*P_k... X_sk may require correction_factor!)

    Attributes:
        theta_sk_over_2 (float): angle to rotate by in radians. Note divided by 2 due to definition of exponentiated
                                 Pauli terms (https://arxiv.org/pdf/1001.3855.pdf)!


    """

    def __init__(self, Pauli_str, Correction_value, dagger=False):

        self.Pauli_str = Pauli_str
        self.dagger = dagger
        self.Correction_value = Correction_value

    def _unitary_(self):

        if self.dagger:

            if self.Pauli_str == 'Z':
                P_mod = cirq.Z._unitary_() * self.Correction_value
                return P_mod
            elif self.Pauli_str == 'Y':
                P_mod = cirq.Y._unitary_() * self.Correction_value
                return P_mod
            elif self.Pauli_str == 'X':
                P_mod = cirq.X._unitary_() * self.Correction_value
                return P_mod
            elif self.Pauli_str == 'I':
                P_mod = cirq.I._unitary_() * self.Correction_value
                return P_mod
            else:
                raise TypeError('not a Pauli operation')
        else:
            if self.Pauli_str == 'Z':
                P_mod = (cirq.Z._unitary_() * self.Correction_value).transpose().conj()
                return P_mod
            elif self.Pauli_str == 'Y':
                P_mod = (cirq.Y._unitary_() * self.Correction_value).transpose().conj()
                return P_mod
            elif self.Pauli_str == 'X':
                P_mod = (cirq.X._unitary_() * self.Correction_value).transpose().conj()
                return P_mod
            elif self.Pauli_str == 'I':
                P_mod = (cirq.I._unitary_() * self.Correction_value).transpose().conj()
                return P_mod
            else:
                raise TypeError('not a Pauli operation')

    def num_qubits(self):
        return 1

    def _circuit_diagram_info_(self, args):
        # NOTE THAT ABOVE term is angle multiplied by constant!!!! V Important to take this into account!
        # Takes into account PauliWord constant.

        if self.dagger:
            return 'PauliMod : {}_dag times {}'.format(self.Pauli_str, self.Correction_value)
        else:
            return 'PauliMod : {} times {}'.format(self.Pauli_str, self.Correction_value)


if __name__ == '__main__':
    Pauli_str = 'Z'
    Correction_value = -1
    dag = False

    test_gate = PauliModified_gate(Pauli_str, Correction_value, dagger=dag)
    circ = test_gate.on(cirq.LineQubit(1))
    print(cirq.Circuit(circ))


class Perform_Modified_PauliWord(cirq.Gate):
    """
    Class to generate cirq circuit as gate that performs a modified PauliWord

    Args:
        LCU_PauliWord_and_cofactor (tuple): Tuple of PauliWord (str) and constant (complex) ... (PauliWord, constant)
        LCU_correction_value (complex):
        dagger (bool): whether to have dagger gate or not

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
                0: ───PauliMod : Y times -1j───
                1: ───PauliMod : Z times -1j───
                2: ───PauliMod : X times -1j───
                3: ───PauliMod : I times -1j───

    """

    def __init__(self, LCU_PauliWord_and_cofactor, LCU_correction_value, dagger):

        self.PauliWord_and_cofactor = LCU_PauliWord_and_cofactor
        self.correction_value = LCU_correction_value
        self.dagger = dagger

    def _decompose_(self, qubits):

        PauliWord = self.PauliWord_and_cofactor[0].split(' ')

        for PauliString in PauliWord:
            qubitOp = PauliString[0]
            qubitNo = int(PauliString[1::])

            yield PauliModified_gate(qubitOp, self.correction_value, dagger=self.dagger).on(qubits[qubitNo])

    def _circuit_diagram_info_(self, args):
        string_list = []
        PauliWord = self.PauliWord_and_cofactor[0].split(' ')

        if self.dagger:
            for _ in range(len(PauliWord)):
                string_list.append(' Pauliword_modified_for_LCU_DAGGER ')
        else:
            for _ in range(len(PauliWord)):
                string_list.append(' Pauliword_modified_for_LCU ')
        return string_list

    def num_qubits(self):
        PauliWord = self.PauliWord_and_cofactor[0].split(' ')
        return len(PauliWord)


if __name__ == '__main__':
    test_case = ('Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Y11', 0.00070859248123462)
    correction_val = (-0 - 1j)
    dag = False

    P_circ_mod = Perform_Modified_PauliWord(test_case, correction_val, dag)

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

    def __init__(self, LCU_term_dict, LCU_correction_dict, dagger, No_control_qubits, No_system_qubits):

        self.LCU_term_dict = LCU_term_dict
        self.LCU_correction_dict = LCU_correction_dict
        self.dagger = dagger
        self.No_control_qubits = No_control_qubits
        self.No_system_qubits = No_system_qubits

    def _decompose_(self, qubits):

        state_index = 0
        for control_state in self.LCU_term_dict:
            control_str = Get_state_as_str(self.No_control_qubits, state_index)
            # using state index rather than control_state value
            # due to how LCU_term is built! (aka missing keys cause problems!)

            control_values = [int(bit) for bit in control_str]

            correction_val = self.LCU_correction_dict[control_state]
            mod_P = self.LCU_term_dict[control_state]
            mod_p_word_gate = Perform_Modified_PauliWord(mod_P, correction_val, self.dagger)

            qubit_list = cirq.LineQubit.range(self.No_system_qubits, self.No_system_qubits + self.No_control_qubits) \
                         + cirq.LineQubit.range(self.No_system_qubits)  # note control qubits first!

            state_index += 1
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


if __name__ == '__main__':
    dag = False
    GATE = LCU_R_gate(tt, correction_list, dag, number_ancilla_qubits, Hamilt.molecule.n_qubits)

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
        LCU_terms, LCU_correction_list = Get_R_linear_comb(anti_commuting_set, S_index, No_system_qubits)

        l1_norm_val, number_ancilla_qubits, G_ancilla_amplitudes = Get_ancilla_amplitudes(LCU_terms)

        alpha_j = Get_state_prep_dict(number_ancilla_qubits, Coefficient_list=G_ancilla_amplitudes)
        state_circ = State_Prep_Circuit(alpha_j)
        ancilla_line_qubits = cirq.LineQubit.range(No_system_qubits, No_system_qubits + number_ancilla_qubits)

        G_prep_circuit = (cirq.Circuit(cirq.decompose_once((state_circ(*ancilla_line_qubits)))))

        # G_prep_circuit.append(cirq.measure(*ancilla_line_qubits)) # add measurement gates to ancilla line

        R_dagger_gate_obj = LCU_R_gate(LCU_terms, LCU_correction_list, True, number_ancilla_qubits, No_system_qubits)
        R_dagger_gate = cirq.Circuit(
            cirq.decompose_once((R_dagger_gate_obj(*cirq.LineQubit.range(R_dagger_gate_obj.num_qubits())))))

        Pauli_S = anti_commuting_set[S_index]

        Pauli_S_circ_obj = Perform_PauliWord(Pauli_S)
        Pauli_S_circ = cirq.Circuit(
            cirq.decompose_once((Pauli_S_circ_obj(
                *cirq.LineQubit.range(Pauli_S_circ_obj.num_qubits())))))

        R_gate_obj = LCU_R_gate(LCU_terms, LCU_correction_list, False, number_ancilla_qubits, No_system_qubits)
        R_gate = cirq.Circuit(
            cirq.decompose_once((R_gate_obj(*cirq.LineQubit.range(R_gate_obj.num_qubits())))))

        measure_gate_obj = Measure_system_and_ancilla(Pauli_S,number_ancilla_qubits)

        measure_circ = cirq.Circuit(
            cirq.decompose_once((measure_gate_obj(*cirq.LineQubit.range(measure_gate_obj.num_qubits())))))

        full_circuit = cirq.Circuit(
            [
                G_prep_circuit.all_operations(),
                *R_gate.all_operations(),
                Pauli_S_circ.all_operations(),
                R_dagger_gate.all_operations(),
                measure_circ.all_operations()
            ]
        )
        return full_circuit


if __name__ == '__main__':
    test_set = anti_commuting_set_stripped[58]
    S_index = 0
    circuit = Complete_LCU_circuit(test_set, Hamilt.molecule.n_qubits, S_index)
    print(circuit)




# LCU GUG method



def ALCU_dict(Full_Ansatz_Q_Circuit, anti_commuting_sets, S_dict,
                                                     n_system, n_ancilla):
    """
     Function that appends Ansatz Quantum Circuit to Pauli perform and measure circuit instance.

    Args:
        Full_Ansatz_Q_Circuit (cirq.circuits.circuit.Circuit): Full cirq Ansatz Q circuit
        anti_commuting_sets (dict): dict of list of tuples containing PauliWord to perform and constant -> (PauliWord, constant)
        S_dict (dict): dict of PauliWordS indices
        n_system (int): number of system qubits
        n_ancilla (int): number of ancilla

    Returns:
        dic_holder (dict): Returns a dictionary of each quantum circuit, with cofactor, PauliWord and cirq Q Circuit

    """

    I_Measure = ['I{}'.format(i) for i in range(n_system)]
    seperator = ' '
    PauliWord_I_only = seperator.join(I_Measure)

    ancilla_list = ['Z{}'.format(int(i)) for i in np.arange(n_system, n_system + n_ancilla)]
    seperator = ' '
    ancilla_str = seperator.join(ancilla_list)

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
            Q_circuit_obj = Perform_PauliWord_and_Measure(PauliString_and_Constant[0])
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
            ALCU_circuit = Complete_LCU_circuit(PauliString_and_Constant, n_system, S_index)

            full_circuit = cirq.Circuit(
                [
                    Full_Ansatz_Q_Circuit,
                    ALCU_circuit
                ]
            )
            temp_d['circuit'] = full_circuit
            temp_d['gamma_l'] = PauliString_and_Constant[S_index][1]
            temp_d['PauliWord'] = PauliString_and_Constant[S_index][0] + ' ' +ancilla_str
            temp_d['LCU'] = True

        dic_holder[key] = temp_d

    return dic_holder


from quchem.Simulating_Quantum_Circuit import *
class ALCU_Simulation_Quantum_Circuit_Dict():
    def __init__(self, circuits_factor_PauliWord_dict, num_shots, n_ancilla):
        self.circuits_factor_PauliWord_dict = circuits_factor_PauliWord_dict
        self.num_shots = num_shots
        self.n_ancilla = n_ancilla

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

    def Get_binary_results_dict(self):
        if self.hist_key_dict is None:
            self.Get_Histkey_dict()

        binary_results_dict = {}
        for key in self.circuits_factor_PauliWord_dict:

            if self.hist_key_dict[key] != '':
                #checks for non identity ciruict
                if self.circuits_factor_PauliWord_dict[key]['LCU'] is False:
                    counter_result = Simulate_Quantum_Circuit(self.circuits_factor_PauliWord_dict[key]['circuit'],
                                                                self.num_shots, self.hist_key_dict[key])
                    binary_results_dict[key] = Return_as_binary(counter_result, self.circuits_factor_PauliWord_dict[key]['PauliWord'])
                else:
                    simulator = cirq.Simulator()
                    LCU_result_dict = {}
                    ancilla_key = ''.join(['0' for _ in range(self.n_ancilla)])
                    n_success_shots=0
                    while n_success_shots != self.num_shots:
                        raw_result = simulator.run(self.circuits_factor_PauliWord_dict[key]['circuit']) #, repetitions=num_shots)
                        hist_result = raw_result.histogram(key=self.hist_key_dict[key])
                        binary_res = Return_as_binary(hist_result, self.circuits_factor_PauliWord_dict[key]['PauliWord'])
                        for state in binary_res:
                            if state[:(-1-self.n_ancilla):-1] == ancilla_key:
                                if state not in LCU_result_dict.keys():
                                    LCU_result_dict[state] = 0
                                LCU_result_dict[state] = LCU_result_dict[state] + binary_res[state]
                                n_success_shots += 1
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

            elif self.circuits_factor_PauliWord_dict[key]['LCU']:
                # remove ancilla results with slice!
                ancilla_removed = {state[0:-1*self.n_ancilla]: counts for state, counts in self.binary_results_dict[key].items()}
                expect_results_dict[key] = expectation_value_by_parity(ancilla_removed)
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