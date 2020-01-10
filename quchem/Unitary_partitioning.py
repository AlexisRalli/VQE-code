import numpy as np

anti_commuting_sets = {
     0: [('I0 I1 I2 I3', (-0.32760818995565577+0j))],
     1: [('Z0 Z1 I2 I3', (0.15660062486143395+0j))],
     2: [('Z0 I1 Z2 I3', (0.10622904488350779+0j))],
     3: [('Z0 I1 I2 Z3', (0.15542669076236065+0j))],
     4: [('I0 Z1 Z2 I3', (0.15542669076236065+0j))],
     5: [('I0 Z1 I2 Z3', (0.10622904488350779+0j))],
     6: [('I0 I1 Z2 Z3', (0.1632676867167479+0j))],
     7: [('Z0 I1 I2 I3', (0.1371657293179602+0j)), ('Y0 X1 X2 Y3', (0.04919764587885283+0j)), ('X0 I1 I2 I3', (0.04919764587885283+0j))], # <- I added this term to check code
     8: [('I0 Z1 I2 I3', (0.1371657293179602+0j)), ('Y0 Y1 X2 X3', (-0.04919764587885283+0j))],
     9: [('I0 I1 Z2 I3', (-0.13036292044009176+0j)),('X0 X1 Y2 Y3', (-0.04919764587885283+0j))],
     10: [('I0 I1 I2 Z3', (-0.13036292044009176+0j)), ('X0 Y1 Y2 X3', (0.04919764587885283+0j))]
}


def Get_beta_j_cofactors(anti_commuting_set):
    """
    Function takes in list of anti-commuting PauliWord tuples (PauliWord, constant)
    and returns the corresponding anti-commuting sets, but with new coefcators that
    obey eq (10) in ArXiv:1908.08067 (sum_j B_j^2 = 1) and an overall correction (gamma_l).

    Args:
        anti_commuting_set (list): A list of Pauliwords, where each entry is a tuple of (PauliWord, constant)

    Returns:
        dict: A dictionary of normalised_anti_commuting_set (key = 'PauliWords') and correction factor (key = 'gamma_l')

    .. code-block:: python
       :emphasize-lines: 4

       from quchem.Unitary_partitioning import *
       Anti_commuting_set = [('I0 Z1 I2 I3', (0.1371657293179602+0j)), ('Y0 Y1 X2 X3', (-0.04919764587885283+0j))]

       Get_beta_j_cofactors(Anti_commuting_set)
       >> {'PauliWords': [  ('I0 Z1 I2 I3', (0.9412848366792171+0j)),
                            ('Y0 Y1 X2 X3', (-0.33761347164735517+0j))
                          ],
          'gamma_l': (0.14572180914107857+0j)}

    """
    normalised_terms = []
    factor = sum([constant ** 2 for PauliWord, constant in anti_commuting_set])
    for PauliWord, constant in anti_commuting_set:
        new_constant = constant/np.sqrt(factor)
        normalised_terms.append((PauliWord, new_constant))

    return {'PauliWords': normalised_terms, 'gamma_l': np.sqrt(factor)}


if __name__ == '__main__':
    vv = Get_beta_j_cofactors(anti_commuting_sets[7])

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
               but in fact the correction term from multiplying all the Paulis. e.g. YX = -1i Z.

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


def Get_X_sk_operators(normalised_anticommuting_set_DICT, S=0): #
    """

    Function takes in a normalised_anti_commuting_set, which is a list of PauliWord tuples (PauliWord, constant),
    and returns each R_sk operator according to eq (11) in ArXiv:1908.08067.

    Args:
        normalised_anticommuting_set_DICT (list): A list of Pauliwords, where each entry is a tuple of (PauliWord, constant)
        S (optional, int) = index for PauliWord_S term. If not stated then takes first entry (index = 0)

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
            Op_list.append({'X_sk': convert_X_sk(X_sk_op),
                            'theta_sk': theta_sk})  # , 'factor': normalised_anti_commuting_sets[key]['factor']})

            beta_S = beta_K*np.sin(theta_sk) + beta_S*np.cos(theta_sk)



        return {'X_sk_theta_sk': Op_list, 'PauliWord_S': (anti_commuting_set[S][0], beta_S), 'gamma_l': normalised_anticommuting_set_DICT['gamma_l']}


if __name__ == '__main__':
    ww = Get_X_sk_operators(vv)


class X_sk_terms():
    def __init__(self,anti_commuting_sets, S_dict=None):
        self.anti_commuting_sets = anti_commuting_sets

        if S_dict is None:
            # makes PauliS always 0th index in anticommuting sets
            self.S_dict={}
            for key in anti_commuting_sets:
                self.S_dict[key] = 0
        else:
            self.S_dict = S_dict

        self.normalised_anti_commuting_sets = None
        self.X_sk_Ops = None

    def Get_normalised_set(self):
        normalised_anti_commuting_sets={}
        for key in self.anti_commuting_sets:
            if len(self.anti_commuting_sets[key]) <= 1:
                continue
            else:
                normalised_anti_commuting_sets[key] = Get_beta_j_cofactors(self.anti_commuting_sets[key])

        self.normalised_anti_commuting_sets = normalised_anti_commuting_sets

    def Get_X_sk_Operators(self):
        if self.normalised_anti_commuting_sets is None:
            self.Get_normalised_set()

        X_sk_Ops={}
        for key in self.normalised_anti_commuting_sets:
            X_sk_Ops[key] = Get_X_sk_operators(self.normalised_anti_commuting_sets[key], S=self.S_dict[key])

        self.X_sk_Ops = X_sk_Ops

    def Get_all_X_sk_operators(self):
        self.Get_normalised_set()
        self.Get_X_sk_Operators()

if __name__ == '__main__':
    ALL_X_SK_TERMS = X_sk_terms(anti_commuting_sets, S_dict={0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0})
    ALL_X_SK_TERMS.Get_all_X_sk_operators()



import cirq

class Change_of_Basis_initial_X_sk(cirq.Gate):
    """
    Class to generate cirq circuit as gate in order to perform: e^(-i theta_sk/2 * X_sk) ... eq (12) arXiv: 1908.08067

    This class generates change of basis in oder to perform PauliWord as a Z terms only
    aka: e^(cofactor * theta_sk/2 * PauliWord_Z_ONLY)



    e.g.:
           X_sk = ('X0 X1 X2 Y3', (1+0j))
    gives:
                0: ───H──────────
                1: ───H──────────
                2: ───H──────────
                3: ───Rx(0.5π)───


    Args:
        X_sk (tuple): A tuple of tuples: (X_sk, constant). Note that constant here is NOT cofactor from Hamiltonian
               but in fact the correction term from multiplying P_s and P_k... e.g. YX = -1i Z.

    Returns
        A cirq circuit object to be used by cirq.Circuit.from_ops

    """

    def __init__(self, X_sk):

        self.X_sk = X_sk

    def _decompose_(self, qubits):

        PauliWord = self.X_sk[0].split(' ')

        for PauliString in PauliWord:
            qubitOp = PauliString[0]
            qubitNo = int(PauliString[1::])

            if qubitOp == 'X':
                yield cirq.H(qubits[qubitNo])
            elif qubitOp == 'Y':
                 yield cirq.Rx(np.pi / 2)(qubits[qubitNo])
            elif qubitOp == 'Z' or 'I':
                continue
            else:
                raise ValueError("Qubit Operation: {} is NOT a Pauli operation".format(qubitOp))

    def _circuit_diagram_info_(self, args):
        Ansatz_basis_change_list = []
        PauliWord = self.X_sk[0].split(' ')
        for i in range(len(PauliWord)):
                Ansatz_basis_change_list.append('Basis_change')
        return Ansatz_basis_change_list

    def num_qubits(self):
        PauliWord = self.X_sk[0].split(' ')
        return len(PauliWord)

if __name__ == '__main__':
    #X_SK_Test = ww[7][0]['X_sk'] # (  ('Z0 I1 I2 I3', (0.8918294488900189+0j)), ('Y0 X1 X2 Y3', (0.3198751585326103+0j))   )
    X_SK_Test = ALL_X_SK_TERMS.X_sk_Ops[7]['X_sk_theta_sk'][0]['X_sk']
    #X_SK_Test = (  ('Z0 I1 I2 I3 I4 I5 I6 I7 I8 I9 X10', (0.8918294488900189+0j)), ('Y0 X1 X2 Y3 I4 I5 I6 I7 I8 I9 Z10', (0.3198751585326103+0j))   )

    Basis_change_circuit = Change_of_Basis_initial_X_sk(X_SK_Test)

    print(cirq.Circuit.from_ops((Basis_change_circuit(*cirq.LineQubit.range(Basis_change_circuit.num_qubits())))))
    print(
        cirq.Circuit.from_ops(cirq.decompose_once((Basis_change_circuit(*cirq.LineQubit.range(Basis_change_circuit.num_qubits()))))))



class Engtangle_initial_X_sk(cirq.Gate):
    """
    Class to generate cirq circuit as gate... which generates CNOT entangling gates between non Idenity PauliWord
    qubits in order to perform PauliWord as a Z terms only for: e^(cofactor * theta/2 * PauliWord_Z_ONLY)
    where change of basis for  e^(-i theta_sk/2 * X_sk) already performed.

    e.g.: X_sk = ('X0 X1 X2 Y3', (1+0j))
        gives :
                    0: ───@───────────
                          │
                    1: ───X───@───────
                              │
                    2: ───────X───@───
                                  │
                    3: ───────────X───

    Args:
        X_sk (tuple): A tuple of tuples: (X_sk, constant). Note that constant here is NOT cofactor from Hamiltonian
               but in fact the correction term from multiplying P_s and P_k... e.g. YX = -1i Z.

    Returns
        A cirq circuit object to be used by cirq.Circuit.from_ops

    """


    def __init__(self, X_sk):

        self.X_sk = X_sk

    def _decompose_(self, qubits):

        PauliWord = self.X_sk[0].split(' ')

        # note identity terms removed here
        qubitNo_qubitOp_list = [(int(PauliString[1::]), PauliString[0]) for PauliString in PauliWord if PauliString[0] != 'I']

        control_qubit = max([qubitNo for qubitNo, qubitOp in qubitNo_qubitOp_list])

        for j in range(len(qubitNo_qubitOp_list)):
            qubitNo = qubitNo_qubitOp_list[j][0]
            #qubitOp = qubitNo_qubitOp_list[j][1]

            if qubitNo < control_qubit:
                qubitNo_NEXT = qubitNo_qubitOp_list[j + 1][0]
                yield cirq.CNOT(qubits[qubitNo], qubits[qubitNo_NEXT])


    def _circuit_diagram_info_(self, args):
        string_list = []
        PauliWord = self.X_sk[0].split(' ')
        for i in range(len(PauliWord)):
                string_list.append('Entangling circuit')
        return string_list


    def num_qubits(self):
        PauliWord = self.X_sk[0].split(' ')
        return len(PauliWord)


if __name__ == '__main__':
    #X_SK_Test = ww[7][0]['X_sk'] # (  ('Z0 I1 I2 I3', (0.8918294488900189+0j)), ('Y0 X1 X2 Y3', (0.3198751585326103+0j))   )
    X_SK_Test = ALL_X_SK_TERMS.X_sk_Ops[7]['X_sk_theta_sk'][0]['X_sk']

    #X_SK_Test = (  ('Z0 I1 I2 I3 I4 I5 I6 I7 I8 I9 X10', (0.8918294488900189+0j)), ('Y0 X1 X2 Y3 I4 I5 I6 I7 I8 I9 Z10', (0.3198751585326103+0j))   )

    Ent_initial = Engtangle_initial_X_sk(X_SK_Test)

    print(cirq.Circuit.from_ops((Ent_initial(*cirq.LineQubit.range(Ent_initial.num_qubits())))))
    print(
        cirq.Circuit.from_ops(cirq.decompose_once((Ent_initial(*cirq.LineQubit.range(Ent_initial.num_qubits()))))))


class My_R_sk_Gate(cirq.SingleQubitGate):

    """
    My_R_sk_Gate class generate R_sk and R_sk dagger quantum circuits as a gate. Gate defined in
    eq (12) arXiv: 1908.08067.

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

    .. code-block:: python
       :emphasize-lines: 6

       from quchem.Unitary_partitioning import *
       import cirq

       X_sk = ('X0 X1 X2 Y3', (1+0j))
       theta_sk =  (0.34438034648829496+0j)
       R_S_DAGGER = My_R_sk_Gate(theta_sk, dagger=True, correction_factor=X_sk[1])

       # example of use with cirq.
       Q_circuit = R_S.on(cirq.LineQubit(1))
       print(cirq.Circuit.from_ops(Q_circuit))

       >> 1: ───R_sk_DAGGER = (0.34438034648829496+0j) rad───
    """

    def __init__(self, theta_sk, dagger=True, correction_factor=1):

        self.theta_sk_over_2 = theta_sk/2
        self.dagger = dagger
        self.correction_factor = correction_factor

    def _unitary_(self):
        # NOTE THAT ABOVE term is angle multiplied by constant!!!! V Important to take this into account!
        # Takes into account PauliWord constant.
        if self.dagger:

            R_sk_dag = np.array([
                        [np.e** (-0.5j * self.theta_sk_over_2 * self.correction_factor), 0],
                        [0, np.e** (+0.5j * self.theta_sk_over_2 * self.correction_factor)]
                    ])
            #R_sk_dag = cirq.Rz(self.theta_sk * self.correction_factor)._unitary_()
            return R_sk_dag
        else:
            R_sk = np.array([
                [np.e ** (+0.5j * self.theta_sk_over_2 * self.correction_factor), 0],
                [0, np.e ** (-0.5j * self.theta_sk_over_2 * self.correction_factor)]
            ])
            #R_sk = (cirq.Rz(self.theta_sk * self.correction_factor)**-1)._unitary_()
            return R_sk

    def num_qubits(self):
        return 1

    def _circuit_diagram_info_(self, args):
        # NOTE THAT ABOVE term is angle multiplied by constant!!!! V Important to take this into account!
        # Takes into account PauliWord constant.

        if self.dagger:
            return 'R_sk_DAGGER = {} rad'.format(self.theta_sk_over_2 * self.correction_factor)
        else:
            return 'R_sk = {} rad'.format(self.theta_sk_over_2 * self.correction_factor)


if __name__ == '__main__':
    X_SK_Test = ALL_X_SK_TERMS.X_sk_Ops[7]['X_sk_theta_sk'][0]['X_sk']
    theta_sk = ALL_X_SK_TERMS.X_sk_Ops[7]['X_sk_theta_sk'][0]['theta_sk']
    R_S = My_R_sk_Gate(theta_sk, dagger=True, correction_factor=X_SK_Test[1])
    w = R_S.on(cirq.LineQubit(1))
    print(cirq.Circuit.from_ops(w))


class R_sk_DAGGER(cirq.Gate):
    """
    Class that uses My_R_sk_Gate class to generate full R_sk OR R_sk dagger quantum circuit as a gate. Gate defined in
    eq (12) arXiv: 1908.08067.

    Note this puts correct correction_factor for X_sk.

    Args:
        theta_sk (float): angle to rotate by in radians.
        dagger (bool): Where to have dagger or non dagger quantum gate
        X_sk (tuple): Tuple of (PauliWord, correction_factor)
                      e.g. if X_sk = ('X0 X1 X2 Y3', (-1+0j)) then it would be -1.


    .. code-block:: python
       :emphasize-lines: 6

       from quchem.Unitary_partitioning import *
       import cirq

       X_sk = ('X0 X1 X2 Y3', (1+0j))
       theta_sk =  (0.34438034648829496+0j)
       Q_circuit = R_sk_DAGGER(X_SK_Test, theta_sk, dagger=True)

       # example of use with cirq.
       print(cirq.Circuit.from_ops(cirq.decompose_once(
         (R_sk_rot_circuit(*cirq.LineQubit.range(R_sk_rot_circuit.num_qubits()))))))

       >> 3: ───R_sk_DAGGER = (0.34438034648829496+0j) rad───
    """
    def __init__(self, X_sk, theta_sk, dagger=True):

        self.X_sk = X_sk
        self.theta_sk = theta_sk
        self.dagger = dagger


    def _decompose_(self, qubits):

        PauliWord = self.X_sk[0].split(' ')

        # note identity terms removed here
        qubitNo_qubitOp_list = [(int(PauliString[1::]), PauliString[0]) for PauliString in PauliWord if PauliString[0] != 'I']

        control_qubit = max([qubitNo for qubitNo, qubitOp in qubitNo_qubitOp_list])

        yield My_R_sk_Gate(self.theta_sk, dagger=self.dagger, correction_factor=self.X_sk[1]).on(qubits[control_qubit])

    def num_qubits(self):
        PauliWord = self.X_sk[0].split(' ')
        return len(PauliWord)

    def _circuit_diagram_info_(self, args):
        string_list = []
        PauliWord = self.X_sk[0].split(' ')
        for i in range(len(PauliWord)):
                string_list.append('R_sk_rotation circuit')
        return string_list

if __name__ == '__main__':
    X_SK_Test = ALL_X_SK_TERMS.X_sk_Ops[7]['X_sk_theta_sk'][0]['X_sk']
    theta_sk = ALL_X_SK_TERMS.X_sk_Ops[7]['X_sk_theta_sk'][0]['theta_sk']
    #X_SK_Test = (  ('Z0 I1 I2 I3 I4 I5 I6 I7 I8 I9 X10', (0.8918294488900189+0j)), ('Y0 X1 X2 Y3 I4 I5 I6 I7 I8 I9 Z10', (0.3198751585326103+0j))   )

    R_sk_rot_circuit = R_sk_DAGGER(X_SK_Test, theta_sk, dagger=True)

    print(cirq.Circuit.from_ops((R_sk_rot_circuit(*cirq.LineQubit.range(R_sk_rot_circuit.num_qubits())))))
    print(
        cirq.Circuit.from_ops(cirq.decompose_once((R_sk_rot_circuit(*cirq.LineQubit.range(R_sk_rot_circuit.num_qubits()))))))



class Engtangle_final_X_sk(cirq.Gate):

    """
    Class to generate cirq circuit as gate... which generates CNOT entangling gates between non Idenity PauliWord
    qubits in order to perform PauliWord as a Z terms only for: e^(cofactor * theta/2 * PauliWord_Z_ONLY)
    where change of basis for  e^(-i theta_sk/2 * X_sk) already performed.

    e.g.: X_sk = ('X0 X1 X2 Y3', (1+0j))
        gives :
                    0: ───────────@───
                                  │
                    1: ───────@───X───
                              │
                    2: ───@───X───────
                          │
                    3: ───X───────────

    Args:
        X_sk (tuple): A tuple of tuples: (X_sk, constant). Note that constant here is NOT cofactor from Hamiltonian
               but in fact the correction term from multiplying P_s and P_k... e.g. YX = -1i Z.

    Returns
        A cirq circuit object to be used by cirq.Circuit.from_ops

    """

    def __init__(self, X_sk):

        self.X_sk = X_sk

    def _decompose_(self, qubits):

        PauliWord = self.X_sk[0].split(' ')

        # note identity terms removed here
        qubitNo_qubitOp_list_REVERSE = [(int(PauliString[1::]), PauliString[0]) for PauliString in PauliWord if PauliString[0] != 'I'][::-1]

        control_qubit = max([qubitNo for qubitNo, qubitOp in qubitNo_qubitOp_list_REVERSE])

        for i in range(len(qubitNo_qubitOp_list_REVERSE)):
            qubitNo, qubitOp = qubitNo_qubitOp_list_REVERSE[i]

            if qubitNo < control_qubit and qubitNo >= 0:
                qubitNo_NEXT = qubitNo_qubitOp_list_REVERSE[i - 1][0]   # note negative here
                yield cirq.CNOT(qubits[qubitNo], qubits[qubitNo_NEXT])


    def _circuit_diagram_info_(self, args):
        string_list = []
        PauliWord = self.X_sk[0].split(' ')
        for i in range(len(PauliWord)):
                string_list.append('Entangling circuit')
        return string_list


    def num_qubits(self):
        PauliWord = self.X_sk[0].split(' ')
        return len(PauliWord)


if __name__ == '__main__':
    X_SK_Test = ALL_X_SK_TERMS.X_sk_Ops[7]['X_sk_theta_sk'][0]['X_sk']
    #X_SK_Test = (  ('Z0 I1 I2 I3 I4 I5 I6 I7 I8 I9 X10', (0.8918294488900189+0j)), ('Y0 X1 X2 Y3 I4 I5 I6 I7 I8 I9 Z10', (0.3198751585326103+0j))   )

    Ent_final = Engtangle_final_X_sk(X_SK_Test)

    print(cirq.Circuit.from_ops((Ent_final(*cirq.LineQubit.range(Ent_final.num_qubits())))))
    print(
        cirq.Circuit.from_ops(cirq.decompose_once((Ent_final(*cirq.LineQubit.range(Ent_final.num_qubits()))))))

class Change_of_Basis_final_X_sk(cirq.Gate):

    """
    Class to generate cirq circuit as gate... which generates CNOT entangling gates between non Idenity PauliWord
    qubits in order to perform PauliWord as a Z terms only for: e^(cofactor * theta/2 * PauliWord_Z_ONLY)
    where change of basis for  e^(-i theta_sk/2 * X_sk) already performed.

    e.g.: X_sk = ('X0 X1 X2 Y3', (1+0j))
        gives :
                    0: ───────────@───
                                  │
                    1: ───────@───X───
                              │
                    2: ───@───X───────
                          │
                    3: ───X───────────

    Args:
        X_sk (tuple): A tuple of tuples: (X_sk, constant). Note that constant here is NOT cofactor from Hamiltonian
               but in fact the correction term from multiplying P_s and P_k... e.g. YX = -1i Z.

    Returns
        A cirq circuit object to be used by cirq.Circuit.from_ops

    """

    def __init__(self, X_sk):

        self.X_sk = X_sk

    def _decompose_(self, qubits):

        PauliWord = self.X_sk[0].split(' ')

        for PauliString in PauliWord:
            qubitOp = PauliString[0]
            qubitNo = int(PauliString[1::])

            if qubitOp == 'X':
                yield cirq.H(qubits[qubitNo])
            elif qubitOp == 'Y':
                 yield cirq.Rx(-np.pi / 2)(qubits[qubitNo])
            elif qubitOp == 'Z' or 'I':
                continue
            else:
                raise ValueError("Qubit Operation: {} is NOT a Pauli operation".format(qubitOp))

    def _circuit_diagram_info_(self, args):
        Ansatz_basis_change_list = []
        PauliWord = self.X_sk[0].split(' ')
        for i in range(len(PauliWord)):
                Ansatz_basis_change_list.append('Basis_change')
        return Ansatz_basis_change_list

    def num_qubits(self):
        PauliWord = self.X_sk[0].split(' ')
        return len(PauliWord)

if __name__ == '__main__':
    X_SK_Test = ALL_X_SK_TERMS.X_sk_Ops[7]['X_sk_theta_sk'][0]['X_sk']
    #X_SK_Test = (  ('Z0 I1 I2 I3 I4 I5 I6 I7 I8 I9 X10', (0.8918294488900189+0j)), ('Y0 X1 X2 Y3 I4 I5 I6 I7 I8 I9 Z10', (0.3198751585326103+0j))   )

    Basis_change_circuit = Change_of_Basis_final_X_sk(X_SK_Test)

    print(cirq.Circuit.from_ops((Basis_change_circuit(*cirq.LineQubit.range(Basis_change_circuit.num_qubits())))))
    print(
        cirq.Circuit.from_ops(cirq.decompose_once((Basis_change_circuit(*cirq.LineQubit.range(Basis_change_circuit.num_qubits()))))))

class R_sk_full_circuit(cirq.Gate):

    """
    Class to generate cirq circuit as gate performing : e^(cofactor * theta * X_sk) OR e^(cofactor * theta * X_sk)^DAGGER

    e.g.: X_SK_Test =('X0 X1 X2 Y3', (1+0j))
          theta_sk = np.pi
          dagger=True

        gives :
                0: ───H──────────@───────────────────────────────────────────────────────────────@───H───────────
                                 │                                                               │
                1: ───H──────────X───@───────────────────────────────────────────────────────@───X───H───────────
                                     │                                                       │
                2: ───H──────────────X───@───────────────────────────────────────────────@───X───────H───────────
                                         │                                               │
                3: ───Rx(0.5π)───────────X───R_sk_DAGGER = (1.5707963267948966+0j) rad───X───────────Rx(-0.5π)───

    Args:
        X_sk (tuple): A tuple of tuples: (X_sk, constant). Note that constant here is NOT cofactor from Hamiltonian
                      but in fact the correction term from multiplying P_s and P_k... e.g. YX = -1i Z.
        theta_sk (float): angle to rotate
        dagger (bool): whether to have daggered or non dagger circuit.

    Returns
        A cirq circuit object to be used by cirq.Circuit.from_ops

    """


    def __init__(self, X_sk, theta_sk, dagger):

        self.X_sk = X_sk
        self.theta_sk = theta_sk
        self.dagger = dagger


    def _decompose_(self, qubits):


        Basis_change_initial_circuit = Change_of_Basis_initial_X_sk(self.X_sk)
        Ent_initial = Engtangle_initial_X_sk(self.X_sk)
        R_sk_rot_circuit = R_sk_DAGGER(self.X_sk, self.theta_sk, dagger=self.dagger)
        Ent_final = Engtangle_final_X_sk(self.X_sk)

        Basis_change_final_circuit = Change_of_Basis_final_X_sk(self.X_sk)

        basis_change_initial_gen = Basis_change_initial_circuit._decompose_(qubits)
        Ent_initial_gen = Ent_initial._decompose_(qubits)
        R_sk_rot_circuit_gen = R_sk_rot_circuit._decompose_(qubits)
        Ent_final_gen = Ent_final._decompose_(qubits)
        basis_change_final_gen = Basis_change_final_circuit._decompose_(qubits)

        list_generators = [basis_change_initial_gen, Ent_initial_gen, R_sk_rot_circuit_gen, Ent_final_gen,
                           basis_change_final_gen]
        yield list_generators



    def _circuit_diagram_info_(self, args):
        string_list = []
        PauliWord = self.X_sk[0].split(' ')
        for i in range(len(PauliWord)):
                string_list.append('R_sk_circuit')
        return string_list

    def num_qubits(self):
        PauliWord = self.X_sk[0].split(' ')
        return len(PauliWord)

if __name__ == '__main__':
    X_SK_Test = ALL_X_SK_TERMS.X_sk_Ops[7]['X_sk_theta_sk'][0]['X_sk']
    theta_sk = ALL_X_SK_TERMS.X_sk_Ops[7]['X_sk_theta_sk'][0]['theta_sk']
    # X_SK_Test = (  ('Z0 I1 I2 I3 I4 I5 I6 I7 I8 I9 X10', (0.8918294488900189+0j)), ('Y0 X1 X2 Y3 I4 I5 I6 I7 I8 I9 Z10', (0.3198751585326103+0j))   )

    R_sk_full = R_sk_full_circuit(X_SK_Test, theta_sk, dagger=True)

    print(cirq.Circuit.from_ops((R_sk_full(*cirq.LineQubit.range(R_sk_full.num_qubits())))))
    print(
        cirq.Circuit.from_ops(
            cirq.decompose_once((R_sk_full(*cirq.LineQubit.range(R_sk_full.num_qubits()))))))



# TODO build R_S operator!


def Get_R_S_operators(X_sk_and_theta_sk, dagger=True):
    """
    Function takes in a dictionary from dictionary output from Get_X_sk_operators function... (dict of dict)

    which is a  dictionary of 'PauliWord_S' yields (PauliWord, correction_factor_due_matrix_multiplication), the
    normalisation correction value 'gamma_l' (complex) and each 'X_sk_theta_sk'... which is a list of
    dictionaries that are defined with 'X_sk' = (PauliWord, correction_factor_due_matrix_multiplication) and
    'theta_sk' is rotational angle in radians. NOTE: each element of X_sk_theta_sk dict is a list of sub
    dictionaries each associated to one sk term.

    and outputs quantum circuits with for each R_s_k operation. Note number of terms in list is same as length of
    X_sk_theta_sk in in input dictionary.


    Args:
        X_sk_and_theta_sk (dict): A dictionary of
        dagger (bool, optional): Whether to have daggered operation or not

    Returns:
        list: A list of quantum dictionaries, containing quantum circuit generators and gamma_l terms.

    .. code-block:: python
       :emphasize-lines: 11

       from quchem.Unitary_partitioning import *
       X_sk_and_theta_sk =
                      {
                         'X_sk_theta_sk': [   {'X_sk': ('X0 X1 X2 Y3', (1+0j)), 'theta_sk': (0.34438034648829496+0j)},
                                                {'X_sk': ('Y0 I1 I2 I3', (-1+0j)), 'theta_sk': (0.325597719954341+0j)}
                                            ],
                         'PauliWord_S': ('Z0 I1 I2 I3', (1+0j)),
                         'gamma_l': (0.1538026463340925+0j)
                      }

       Get_R_S_operators(input, dagger=True)
       >> [
            {'q_circuit': <__main__.R_sk_full_circuit at 0x7fb7dc2839b0>,
             'gamma_l': (0.1538026463340925+0j)},

            {'q_circuit': <__main__.R_sk_full_circuit at 0x7fb7dc2837f0>,
            'gamma_l': (0.1538026463340925+0j)}
          ]

    """

    list_quantum_circuits_and_gammal = []
    for terms in X_sk_and_theta_sk['X_sk_theta_sk']:
        R_s_k_circuit_instance = R_sk_full_circuit(terms['X_sk'], terms['theta_sk'], dagger=dagger)

        correction_factor = X_sk_and_theta_sk['gamma_l']

        list_quantum_circuits_and_gammal.append({'q_circuit': R_s_k_circuit_instance, 'gamma_l': correction_factor})

    return list_quantum_circuits_and_gammal


if __name__ == '__main__':
    X_sk_and_theta_sk = ALL_X_SK_TERMS.X_sk_Ops[7]
    qq = Get_R_S_operators(X_sk_and_theta_sk, dagger=True)
    print(cirq.Circuit.from_ops(
        (qq[0]['q_circuit'](*cirq.LineQubit.range(qq[0]['q_circuit'].num_qubits())))))
    print(cirq.Circuit.from_ops(cirq.decompose_once(
        (qq[0]['q_circuit'](*cirq.LineQubit.range(qq[0]['q_circuit'].num_qubits()))))))


from quchem.quantum_circuit_functions import *

class UnitaryPartition(X_sk_terms):
    """
    Class that takes in dictionary of anti_commuting_sets of PauliStrings, full ansatz Q circuit and dictionary of
    Pauli_S indices (index of which term to take as PauliS in anti_commuting_sets dict).

    Args:
        anti_commuting_sets (dict):
        S_dict (): Dictionary with keys corresponding to anti_commuting_sets (dict) with values
                    associated with which term to take as the Pauli_S term. If S_dict=None then takes 0th index
                    of all anti_commuting sets.
        full_anstaz_circuit (): cirq Ansatz Q circuit.

    Attributes:
        X_sk_Ops ():
        normalised_anti_commuting_sets ():
        circuits_and_constants ():

    """


    def __init__(self, anti_commuting_sets, full_anstaz_circuit, S_dict=None):
        self.anti_commuting_sets = anti_commuting_sets
        super().__init__(anti_commuting_sets, S_dict=S_dict)
        self.Get_all_X_sk_operators()
        self.full_anstaz_circuit = full_anstaz_circuit

    def Get_Quantum_circuits_and_constants(self):

        circuits_and_constants = {}
        ansatz_circuit = list(self.full_anstaz_circuit.all_operations())
        for key in self.anti_commuting_sets:
            if key not in self.X_sk_Ops:
                PauliWord_constant = self.anti_commuting_sets[key]

                Pauli_circuit_object = Perform_PauliWord_and_Measure(*PauliWord_constant)
                q_circuit_Pauliword = cirq.Circuit.from_ops(
                    cirq.decompose_once(
                        (Pauli_circuit_object(*cirq.LineQubit.range(Pauli_circuit_object.num_qubits())))))
                circuit_ops = list(q_circuit_Pauliword.all_operations())

                if circuit_ops == []:
                    # deals with identity only circuit
                    circuits_and_constants[key] = {'circuit': None,
                                                   'gamma_l': PauliWord_constant[0][1], 'PauliWord': PauliWord_constant[0][0]}
                else:
                    full_circuit = cirq.Circuit.from_ops(
                        [
                            *ansatz_circuit,
                            *circuit_ops
                        ])

                    circuits_and_constants[key] = {'circuit': full_circuit,
                                                   'gamma_l': PauliWord_constant[0][1], 'PauliWord': PauliWord_constant[0][0]}
            else:

                R_S_DAGGER_operators = Get_R_S_operators(self.X_sk_Ops[key], dagger=True)

                term_reduction_circuits_first = [cirq.decompose_once(
                    (term['q_circuit'](*cirq.LineQubit.range(term['q_circuit'].num_qubits())))) for term in R_S_DAGGER_operators]

                Pauliword_S = self.X_sk_Ops[key]['PauliWord_S']
                q_circuit_Pauliword_S_object = Perform_PauliWord(Pauliword_S)

                q_circuit_Pauliword_S = cirq.Circuit.from_ops(
                    cirq.decompose_once((q_circuit_Pauliword_S_object(
                        *cirq.LineQubit.range(q_circuit_Pauliword_S_object.num_qubits())))))


                R_S_operators = Get_R_S_operators(self.X_sk_Ops[key], dagger=False)

                term_reduction_circuits_LAST = [cirq.decompose_once(
                    (term['q_circuit'](*cirq.LineQubit.range(term['q_circuit'].num_qubits())))) for term in R_S_operators]

                q_circuit_change_basis_and_measure = Change_Basis_and_Measure_PauliWord(Pauliword_S)

                q_circuit_Pauliword_S_change_basis_and_measure = cirq.Circuit.from_ops(
                    cirq.decompose_once(
                        (q_circuit_change_basis_and_measure(
                            *cirq.LineQubit.range(q_circuit_change_basis_and_measure.num_qubits())))))

                full_circuit = cirq.Circuit.from_ops(
                    [
                        *ansatz_circuit,
                        *term_reduction_circuits_first,
                        *q_circuit_Pauliword_S.all_operations(),
                        *term_reduction_circuits_LAST,
                        *q_circuit_Pauliword_S_change_basis_and_measure.all_operations()
                    ]
                )

                # TODO not sure if this part of code is correct!!!
                #iX_sk_constant = np.prod([X_sk['X_sk'][1] for X_sk in self.X_sk_Ops[key]['X_sk_theta_sk']])
                # iX_sk_constant=1

                circuits_and_constants[key] = {'circuit': full_circuit,
                                               'gamma_l': self.X_sk_Ops[key]['gamma_l']*Pauliword_S[1], # note multiplying by factor in front of pauliword!
                                               'PauliWord': Pauliword_S[0]}

            self.circuits_and_constants = circuits_and_constants


if __name__ == '__main__':
    full_anstaz_circuit = [cirq.X.on(cirq.LineQubit(2)),
                           cirq.X.on(cirq.LineQubit(3)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(0)),
                           cirq.H.on(cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.Rz(np.pi * 0.0).on(cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(0)),
                           cirq.H.on(cirq.LineQubit(2)),
                           cirq.H.on(cirq.LineQubit(0)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.Rz(np.pi * 0.0).on(cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.H.on(cirq.LineQubit(0)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(2)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(1)),
                           cirq.H.on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.Rz(np.pi * 0.3183098861837907).on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(1)),
                           cirq.H.on(cirq.LineQubit(3)),
                           cirq.H.on(cirq.LineQubit(1)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.Rz(np.pi * -0.3183098861837907).on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.H.on(cirq.LineQubit(1)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(3)),
                           cirq.H.on(cirq.LineQubit(0)),
                           cirq.H.on(cirq.LineQubit(1)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(2)),
                           cirq.H.on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.Rz(np.pi * 0.6366197723675814).on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.H.on(cirq.LineQubit(0)),
                           cirq.H.on(cirq.LineQubit(1)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(2)),
                           cirq.H.on(cirq.LineQubit(3)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(0)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(1)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(2)),
                           cirq.H.on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.Rz(np.pi * -0.6366197723675814).on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(0)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(1)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(2)),
                           cirq.H.on(cirq.LineQubit(3)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(0)),
                           cirq.H.on(cirq.LineQubit(1)),
                           cirq.H.on(cirq.LineQubit(2)),
                           cirq.H.on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.Rz(np.pi * -0.6366197723675814).on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(0)),
                           cirq.H.on(cirq.LineQubit(1)),
                           cirq.H.on(cirq.LineQubit(2)),
                           cirq.H.on(cirq.LineQubit(3)),
                           cirq.H.on(cirq.LineQubit(0)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(1)),
                           cirq.H.on(cirq.LineQubit(2)),
                           cirq.H.on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.Rz(np.pi * -0.6366197723675814).on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.H.on(cirq.LineQubit(0)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(1)),
                           cirq.H.on(cirq.LineQubit(2)),
                           cirq.H.on(cirq.LineQubit(3)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(0)),
                           cirq.H.on(cirq.LineQubit(1)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(2)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.Rz(np.pi * 0.6366197723675814).on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(0)),
                           cirq.H.on(cirq.LineQubit(1)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(2)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(3)),
                           cirq.H.on(cirq.LineQubit(0)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(1)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(2)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.Rz(np.pi * 0.6366197723675814).on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.H.on(cirq.LineQubit(0)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(1)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(2)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(3)),
                           cirq.H.on(cirq.LineQubit(0)),
                           cirq.H.on(cirq.LineQubit(1)),
                           cirq.H.on(cirq.LineQubit(2)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.Rz(np.pi * 0.6366197723675814).on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.H.on(cirq.LineQubit(0)),
                           cirq.H.on(cirq.LineQubit(1)),
                           cirq.H.on(cirq.LineQubit(2)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(3)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(0)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(1)),
                           cirq.H.on(cirq.LineQubit(2)),
                           cirq.Rx(np.pi * 0.5).on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.Rz(np.pi * -0.6366197723675814).on(cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(2), cirq.LineQubit(3)),
                           cirq.CNOT.on(cirq.LineQubit(1), cirq.LineQubit(2)),
                           cirq.CNOT.on(cirq.LineQubit(0), cirq.LineQubit(1)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(0)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(1)),
                           cirq.H.on(cirq.LineQubit(2)),
                           cirq.Rx(np.pi * -0.5).on(cirq.LineQubit(3))]
    ANSATZ = cirq.Circuit.from_ops(*full_anstaz_circuit)
    zz = UnitaryPartition(anti_commuting_sets, ANSATZ, S_dict=None)
    zz.Get_Quantum_circuits_and_constants()
    zz.circuits_and_constants

