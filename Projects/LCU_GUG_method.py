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
     """
    # TODO need to absorb complex phase into operator!!!!!!!
    Pauli_S = anti_commuting_set[S_index][0]

    seperator = ' '
    I_term = seperator.join(['I{}'.format(i) for i in range(no_qubits)])

    THETA = np.arccos(anti_commuting_set[S_index][1])

    const=np.sin(THETA/2)

    terms=[ (I_term, np.cos(THETA/2)) ]
    correction_list=[1]
    for i in range(len(anti_commuting_set)):
        if i!=S_index:
            P_term = convert(anti_commuting_set[i], Pauli_S)

            factor = P_term[1]*const
            sign_correction=1
            complex_correction=1

            # if factor.real <0:
            #     sign_correction=factor.real
            if np.iscomplex(factor) and factor.imag!=0:
                if factor.imag<0:
                    correction_list.append(-1j)
                    terms.append((P_term[0], np.abs(factor.imag)))
                else:
                    correction_list.append(1j)
                    terms.append((P_term[0], factor.imag))

            elif factor.real <0:
                correction_list.append(-1)
                terms.append((P_term[0], np.abs(factor.real)))
                sign_correction =factor.real
            else:
                correction_list.append(1)
                terms.append((P_term[0], factor.real))

    return terms, correction_list

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


