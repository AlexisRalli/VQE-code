from quchem.Hamiltonian_Generator_Functions import *

### Variable Parameters
Molecule = 'LiH'#LiH'
geometry = None
num_shots = 10000
HF_occ_index = [0,1,2] #[0, 1,2] # for occupied_orbitals_index_list
#######

### Get Hamiltonian
Hamilt = Hamiltonian(Molecule,
                     run_scf=1, run_mp2=1, run_cisd=1, run_ccsd=1, run_fci=1,
                     basis='sto-3g',
                     multiplicity=1,
                     geometry=geometry)  # normally None!

Hamilt.Get_Molecular_Hamiltonian()
SQ_CC_ops, THETA_params = Hamilt.Get_ia_and_ijab_terms(Coupled_cluser_param=True)
#print('UCC operations: ', SQ_CC_ops)

HF_transformations = Hamiltonian_Transforms(Hamilt.MolecularHamiltonian, SQ_CC_ops, Hamilt.molecule.n_qubits)

QubitHam = HF_transformations.Get_Qubit_Hamiltonian_JW()
#print('Qubit Hamiltonian: ', QubitHam)
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

    PauliWord_k = beta_k_Pk[0].split(' ')
    PauliWord_s = Ps.split(' ')

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

    return (new_PauliWord, new_constant_SIGN* beta_k_Pk[1])

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

    const=np.sin(THETA/2)

    LCU_terms={0: (I_term, np.cos(THETA/2)) }
    LCU_correction_dict={0: 1}
    for i in range(len(anti_commuting_set)):
        if i!=S_index:
            P_term = convert(anti_commuting_set[i], Pauli_S)

            factor = P_term[1]*const

            # if factor.real <0:
            #     sign_correction=factor.real
            if np.iscomplex(factor) and factor.imag!=0:
                if factor.imag<0:
                    LCU_correction_dict[i+1] = -1j
                    LCU_terms[i+1] = (P_term[0], np.abs(factor.imag))
                else:
                    LCU_correction_dict[i+1] = 1j
                    LCU_terms[i+1] = (P_term[0], factor.imag)

             elif factor.real <0:
                 LCU_correction_dict[i + 1] = -1
                 LCU_terms[i + 1] = (P_term[0], np.abs(factor.real))
            else:
                LCU_correction_dict[i + 1] = 1
                LCU_terms[i + 1] = (P_term[0], factor.real)
    return LCU_terms, LCU_correction_dict

test_set = anti_commuting_set_stripped[6]
tt, correction_list = Get_R_linear_comb(test_set, 0, Hamilt.molecule.n_qubits)

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

def Get_ancilla_amplitudes(anti_commuting_set):
    """
    Takes in an anti_commuting set and returns l1 norm, number of ancilla qubits and amplitudes required
    on ancilla bits.

    Note that anti_commuting_set = H_s =  ∑_{j=1}^{d} a_{j} U_{j}

    where ||a_vec||_{1} = ∑_{j=1}^{d} |a_{j}| # this is the l1 norm

    need all a_{j}>=0 (complex and negative sign can be absorbed by U_{j}

    Args:
        anti_commuting_set (list):
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
    l1_norm = sum([np.abs(j) for _, j in anti_commuting_set])
    number_ancilla_qubits = int(np.ceil(np.log2(len(anti_commuting_set))))  # note round up with np.ceil

    l1_normalised_amp = [np.abs(amp) / l1_norm for _, amp in anti_commuting_set]
    G_ancilla_amplitudes = [np.sqrt(amp) for amp in l1_normalised_amp]

    if len(G_ancilla_amplitudes) < 2 ** number_ancilla_qubits:
        G_ancilla_amplitudes += [0 for _ in range(int(2 ** number_ancilla_qubits - len(G_ancilla_amplitudes)))]

    return l1_norm, number_ancilla_qubits, G_ancilla_amplitudes

l1_norm_val,number_ancilla_qubits,G_ancilla_amplitudes = Get_ancilla_amplitudes(tt)

## Build state:
from quchem.LCU_method import *

ancilla_line_qubits = cirq.LineQubit.range(Hamilt.molecule.n_qubits, Hamilt.molecule.n_qubits+number_ancilla_qubits)

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
qubits_to_measure = (cirq.LineQubit(q_No) for q_No in range(Hamilt.molecule.n_qubits, Hamilt.molecule.n_qubits+number_ancilla_qubits))
result = simulator.simulate(circuit, qubit_order=qubits_to_measure)
print(np.around(result.final_state, 3))
print('expected state amplitudes:', G_ancilla_amplitudes)


###

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
                P_mod =  cirq.I._unitary_() * self.Correction_value
                return P_mod
            else:
                raise TypeError('not a Pauli operation')
        else:
            if self.Pauli_str == 'Z':
                P_mod = (cirq.Z._unitary_()*self.Correction_value).transpose().conj()
                return P_mod
            elif self.Pauli_str == 'Y':
                P_mod = (cirq.Y._unitary_()*self.Correction_value).transpose().conj()
                return P_mod
            elif self.Pauli_str == 'X':
                P_mod = (cirq.X._unitary_()*self.Correction_value).transpose().conj()
                return P_mod
            elif self.Pauli_str == 'I':
                P_mod =  (cirq.I._unitary_()*self.Correction_value).transpose().conj()
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
    Correction_value=-1
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
    test_case =  ('Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Y11', 0.00070859248123462)
    correction_val = (-0-1j)
    dag = False

    P_circ_mod = Perform_Modified_PauliWord(test_case, correction_val, dag)

    print(cirq.Circuit((P_circ_mod(*cirq.LineQubit.range(P_circ_mod.num_qubits())))))
    print(
        cirq.Circuit(
            cirq.decompose_once((P_circ_mod(*cirq.LineQubit.range(P_circ_mod.num_qubits()))))))


from quchem.LCU_method import Get_state_as_str
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

        for control_state in self.LCU_term_dict:
            control_str = Get_state_as_str(self.No_control_qubits, control_state)
            control_values = [int(bit) for bit in control_str]

            correction_val = self.LCU_correction_dict[control_state]
            mod_P =  self.LCU_term_dict[control_state]
            mod_p_word_gate = Perform_Modified_PauliWord(mod_P, correction_val, self.dagger)

            qubit_list = cirq.LineQubit.range(self.No_system_qubits, self.No_system_qubits+self.No_control_qubits) \
                         + cirq.LineQubit.range(self.No_system_qubits) # note control qubits first!

            yield mod_p_word_gate.controlled(num_controls=self.No_control_qubits, control_values=control_values).on(*qubit_list) # *qubit_list

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
    GATE =  LCU_R_gate(tt, correction_list, dag, number_ancilla_qubits, Hamilt.molecule.n_qubits)

    print(cirq.Circuit((GATE(*cirq.LineQubit.range(GATE.num_qubits())))))
    print(
        cirq.Circuit(
            cirq.decompose_once((GATE(*cirq.LineQubit.range(GATE.num_qubits()))))))


from quchem.quantum_circuit_functions import *

def Complete_LCU_circuit(anti_commuting_set, No_system_qubits, S_index):
    """
     Descript

    Args:
        gg

    Returns:
        full_circuit (cirq.circuits.circuit.Circuit):

    """

    l1_norm_val, number_ancilla_qubits, G_ancilla_amplitudes = Get_ancilla_amplitudes(anti_commuting_set)
    alpha_j = Get_state_prep_dict(number_ancilla_qubits, Coefficient_list=G_ancilla_amplitudes)
    state_circ = State_Prep_Circuit(alpha_j)
    ancilla_line_qubits = cirq.LineQubit.range(No_system_qubits, No_system_qubits + number_ancilla_qubits)

    G_prep_circuit = (cirq.Circuit(cirq.decompose_once((state_circ(*ancilla_line_qubits)))))

    # G_prep_circuit.append(cirq.measure(*ancilla_line_qubits)) # add measurement gates to ancilla line

    LCU_terms, LCU_correction_list = Get_R_linear_comb(test_set, S_index, Hamilt.molecule.n_qubits)

    R_dagger_gate_obj =  LCU_R_gate(LCU_terms, LCU_correction_list, True, number_ancilla_qubits,No_system_qubits)
    R_dagger_gate = cirq.Circuit(
        cirq.decompose_once((R_dagger_gate_obj(*cirq.LineQubit.range(R_dagger_gate_obj.num_qubits())))))

    Pauli_S = anti_commuting_set[S_index]

    Pauli_S_circ_obj = Perform_PauliWord(Pauli_S)
    Pauli_S_circ = cirq.Circuit(
        cirq.decompose_once((Pauli_S_circ_obj(
            *cirq.LineQubit.range(Pauli_S_circ_obj.num_qubits())))))

    R_gate_obj= LCU_R_gate(LCU_terms, LCU_correction_list, False, number_ancilla_qubits, No_system_qubits)
    R_gate = cirq.Circuit(
        cirq.decompose_once((R_gate_obj(*cirq.LineQubit.range(R_gate_obj.num_qubits())))))


    full_circuit = cirq.Circuit(
       [
           G_prep_circuit.all_operations(),
           *R_gate.all_operations(),
           Pauli_S_circ.all_operations(),
           R_dagger_gate.all_operations()
       ]
    )
    return full_circuit
if __name__ == '__main__':
    test_set = anti_commuting_set_stripped[6]
    S_index=0
    circuit = Complete_LCU_circuit(test_set, Hamilt.molecule.n_qubits, S_index)
    print(circuit)