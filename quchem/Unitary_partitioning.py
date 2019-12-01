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
    Function takes in anti_commuting_sets and returns anti-commuting sets, but with new coefcators that
    obey eq (10) in ArXiv:1908.08067 (sum_j B_j^2 = 1)

    Output is a new dictionary, with PauliWords and new constant terms... in other part is correction factor!

    :param anti_commuting_set: A dictionary of anti-commuting sets, each term inside is a tuple of (PauliWord, Constant)
    :type anti_commuting_set: list
    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: New dictionary of anti_commuting_sets, with new constants in Tuples.
    :rtype: dict

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
    converts i P_s P_k of X_sk into one term!
    (Note that P_k is performed first followed by P_s)

    :param X_sk:

    e.g. : (
              ('Z0 I1 I2 I3', (0.8918294488900189+0j)), # P_s
              ('Y0 X1 X2 Y3', (0.3198751585326103+0j))  # P_k
            )

    :return: Returns a list of tuples containing  (new PauliString, new constant)
    :rtype: list

    e.g. ('X0 X1 X2 Y3', (0.28527408634774526+0j))

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

# def Get_X_sk_operators(normalised_anticommuting_set_DICT, S=0): # TODO write function to select 'best' S term!
#     """
#     Function takes in normalised_anti_commuting_set and gets each R_sk operator according to
#     eq (11) in ArXiv:1908.08067.
#
#     Output is a new dictionary, with PauliWords and new constant terms... in other part is correction factor!
#     NOTE S is fixed here!!!
#
#     :param normalised_anticommuting_set_DICT: A dictionary containing a list of tuples: (PauliWord, normalised_Constant)
#                                             AND normalisation correction factor.
#     :type normalised_anticommuting_set_DICT: dict
#
#     e.g.
#             {
#                 'PauliWords': [   ('Z0 I1 I2 I3', (0.8918294488900189+0j)),
#                                   ('Y0 X1 X2 Y3', (0.3198751585326103+0j)),
#                                   ('X0 I1 I2 I3', (0.3198751585326103+0j))   ],
#                 'gamma_l': (0.1538026463340925+0j)
#             }
#
#     :param S: Index of s in R_sk operator. Note that default is zero. TODO can automate this choice!
#     :type S: int
#     ...
#     :raises [ErrorType]: [ErrorDescription]
#     ...
#     :return: A dictionary containing each X_sk operators, (as a list) with associated
#             theta_sk value, the PauliWord_S and the correction factor gamma_l
#
#     {
#     'X_sk_theta_sk':    [     {'X_sk': ('X0 X1 X2 Y3', (0.28527408634774526+0j)),
#                               'theta_sk': (0.34438034648829496+0j)},
#
#                              {'X_sk': ('Y0 I1 I2 I3', (-0.28527408634774526+0j)),
#                                 'theta_sk': (0.3423076794345934+0j)}       ],
#
#      'PauliWord_S': ('Z0 I1 I2 I3', (1+0j)),
#      'gamma_l': (0.1538026463340925+0j)
#      }
#
#
#     NOTE: each element of the outermost dict is a list of sub dictionaries - each associated to one sk term.
#
#     :rtype: dict
#     """
#
#     anti_commuting_set = normalised_anticommuting_set_DICT['PauliWords']
#
#     if len(anti_commuting_set) > 1:
#
#         k_indexes = [index for index in range(len(anti_commuting_set)) if
#                    index != S]
#
#         Op_list = []
#         beta_S = anti_commuting_set[S][1]
#
#         beta_j_sum = 0
#         for k in k_indexes:
#             X_sk_op =(anti_commuting_set[S], anti_commuting_set[k])
#
#             beta_K = anti_commuting_set[k][1]
#             if beta_j_sum == 0:
#                 tan_theta_sk = beta_K / np.sqrt(beta_S**2)
#                 beta_j_sum += beta_K**2
#             else:
#                 tan_theta_sk = beta_K / np.sqrt(beta_S**2 + beta_j_sum**2)# B_k/B_s
#                 beta_j_sum += beta_K**2
#
#             theta_sk = np.arctan(tan_theta_sk)
#
#             Op_list.append({'X_sk': convert_X_sk(X_sk_op), 'theta_sk': theta_sk})#, 'factor': normalised_anti_commuting_sets[key]['factor']})
#
#         new_beta_S = np.sqrt(beta_j_sum + anti_commuting_set[S][1]**2)
#
#         return {'X_sk_theta_sk': Op_list, 'PauliWord_S': (anti_commuting_set[S][0], new_beta_S), 'gamma_l': normalised_anticommuting_set_DICT['gamma_l']}
def Get_X_sk_operators(normalised_anticommuting_set_DICT, S=0): # TODO write function to select 'best' S term!
    """
    Function takes in normalised_anti_commuting_set and gets each R_sk operator according to
    eq (11) in ArXiv:1908.08067.

    Output is a new dictionary, with PauliWords and new constant terms... in other part is correction factor!
    NOTE S is fixed here!!!

    :param normalised_anticommuting_set_DICT: A dictionary containing a list of tuples: (PauliWord, normalised_Constant)
                                            AND normalisation correction factor.
    :type normalised_anticommuting_set_DICT: dict

    e.g.
            {
                'PauliWords': [   ('Z0 I1 I2 I3', (0.8918294488900189+0j)),
                                  ('Y0 X1 X2 Y3', (0.3198751585326103+0j)),
                                  ('X0 I1 I2 I3', (0.3198751585326103+0j))   ],
                'gamma_l': (0.1538026463340925+0j)
            }

    :param S: Index of s in R_sk operator. Note that default is zero. TODO can automate this choice!
    :type S: int
    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: A dictionary containing each X_sk operators, (as a list) with associated
            theta_sk value, the PauliWord_S and the correction factor gamma_l

    {
    'X_sk_theta_sk':    [     {'X_sk': ('X0 X1 X2 Y3', (0.28527408634774526+0j)),
                              'theta_sk': (0.34438034648829496+0j)},

                             {'X_sk': ('Y0 I1 I2 I3', (-0.28527408634774526+0j)),
                                'theta_sk': (0.3423076794345934+0j)}       ],

     'PauliWord_S': ('Z0 I1 I2 I3', (1+0j)),
     'gamma_l': (0.1538026463340925+0j)
     }


    NOTE: each element of the outermost dict is a list of sub dictionaries - each associated to one sk term.

    :rtype: dict
    """

    anti_commuting_set = normalised_anticommuting_set_DICT['PauliWords']

    if len(anti_commuting_set) > 1:

        k_indexes = [index for index in range(len(anti_commuting_set)) if
                   index != S]

        Op_list = []
        beta_S = anti_commuting_set[S][1]

        beta_j_sum = 0
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
    def __init__(self,anti_commuting_sets, S=0):
        self.anti_commuting_sets = anti_commuting_sets
        self.S = S

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
        if self.normalised_anti_commuting_sets == None:
            self.Get_normalised_set()

        X_sk_Ops={}
        for key in self.normalised_anti_commuting_sets:
            X_sk_Ops[key] = Get_X_sk_operators(self.normalised_anti_commuting_sets[key], S=self.S)

        self.X_sk_Ops = X_sk_Ops

    def Get_all_X_sk_operators(self):
        self.Get_normalised_set()
        self.Get_X_sk_Operators()

if __name__ == '__main__':
    ALL_X_SK_TERMS = X_sk_terms(anti_commuting_sets, S=0)
    ALL_X_SK_TERMS.Get_all_X_sk_operators()



import cirq

class Change_of_Basis_initial_X_sk(cirq.Gate):
    def __init__(self, X_sk):
        """
         Circuit to perform change of basis in order to perform: e^(-i theta_sk/2 X_sk) ... eq (12) arXiv: 1908.08067

        :param X_sk: A tuple of tuples: ((PauliWord P_s, constant), (PauliWord P_k)). note have P_s and P_k.
        :type X_sk: tuple

        e.g.: X_sk =
        e.g. ('X0 X1 X2 Y3', -0.28527408634774526j)

        Then build circuit!

        ...
        :raises [ErrorType]: [ErrorDescription]
        ...
        :return: A circuit object to be used by cirq.Circuit.from_ops
        :rtype: class
       """
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
    def __init__(self, X_sk):
        """
         blah

        ...
        :raises [ErrorType]: [ErrorDescription]
        ...
        :return: A circuit object to be used by cirq.Circuit.from_ops
        :rtype: class
       """
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
    def __init__(self, theta_sk, dagger=True, correction_factor=1):
        """""
    Circuit to build a R_sk^DAGGER gate ... eq (12) arXiv: 1908.08067

    :param theta_sk: Value of theta_sk angle.
    :type theta_sk: float

    NOTE that for iPsPk term = X_sk = ('X0 X1 X2 Y3', -0.28527408634774526j)
                                                        ^^^^^^ this is the correction factor!

    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: A circuit object to be used by cirq.Circuit.from_ops
    :rtype: class
        """
        self.theta_sk = theta_sk
        self.dagger = dagger
        self.correction_factor = correction_factor

    def _unitary_(self):
        # NOTE THAT ABOVE term is angle multiplied by constant!!!! V Important to take this into account!
        # Takes into account PauliWord constant.
        if self.dagger:

            R_sk_dag = np.array([
                        [np.e** (-0.5j * self.theta_sk * self.correction_factor), 0],
                        [0, np.e** (+0.5j * self.theta_sk * self.correction_factor)]
                    ])
            #R_sk_dag = cirq.Rz(self.theta_sk * self.correction_factor)._unitary_()
            return R_sk_dag
        else:
            R_sk = np.array([
                [np.e ** (+0.5j * self.theta_sk * self.correction_factor), 0],
                [0, np.e ** (-0.5j * self.theta_sk * self.correction_factor)]
            ])
            #R_sk = (cirq.Rz(self.theta_sk * self.correction_factor)**-1)._unitary_()
            return R_sk

    def num_qubits(self):
        return 1

    def _circuit_diagram_info_(self, args):
        # NOTE THAT ABOVE term is angle multiplied by constant!!!! V Important to take this into account!
        # Takes into account PauliWord constant.

        if self.dagger:
            return 'R_sk_DAGGER = {} rad'.format(self.theta_sk)
        else:
            return 'R_sk = {} rad'.format(self.theta_sk)


# class My_R_sk_Gate(cirq.SingleQubitGate):
#     def __init__(self, theta_sk, dagger=True, correction_factor=1):
#         """""
#     Circuit to build a R_sk^DAGGER gate ... eq (12) arXiv: 1908.08067
#
#     :param theta_sk: Value of theta_sk angle.
#     :type theta_sk: float
#
#     NOTE that for iPsPk term = X_sk = ('X0 X1 X2 Y3', -0.28527408634774526j)
#                                                         ^^^^^^ this is the correction factor!
#
#     ...
#     :raises [ErrorType]: [ErrorDescription]
#     ...
#     :return: A circuit object to be used by cirq.Circuit.from_ops
#     :rtype: class
#         """
#         self.theta_sk = theta_sk
#         self.dagger = dagger
#         self.correction_factor = correction_factor
#
#     def _unitary_(self):
#         full_exponent_term = self.correction_factor * self.theta_sk
#         # NOTE THAT ABOVE term is angle multiplied by constant!!!! V Important to take this into account!
#         # Takes into account PauliWord constant.
#         if self.dagger:
#             # R_sk_dag = np.array([
#             #             [np.e** (+0.5j * full_exponent_term), 0],
#             #             [0, np.e** (-0.5j * full_exponent_term)]
#             #         ])
#             R_sk_dag = cirq.Rz(full_exponent_term)
#             return R_sk_dag._unitary_()
#         else:
#             # R_sk = np.array([
#             #     [np.e ** (-0.5j * full_exponent_term), 0],
#             #     [0, np.e ** (+0.5j * full_exponent_term)]
#             # ])
#             R_sk = cirq.Rz(full_exponent_term)**-1
#             return R_sk._unitary_()
#
#     def num_qubits(self):
#         return 1
#
#     def _circuit_diagram_info_(self, args):
#         full_exponent_term = self.correction_factor * self.theta_sk
#         # NOTE THAT ABOVE term is angle multiplied by constant!!!! V Important to take this into account!
#         # Takes into account PauliWord constant.
#
#         if self.dagger:
#             return 'R_sk_DAGGER = {} rad'.format(full_exponent_term)
#         else:
#             return 'R_sk = {} rad'.format(full_exponent_term)

if __name__ == '__main__':
    X_SK_Test = ALL_X_SK_TERMS.X_sk_Ops[7]['X_sk_theta_sk'][0]['X_sk']
    theta_sk = ALL_X_SK_TERMS.X_sk_Ops[7]['X_sk_theta_sk'][0]['theta_sk']
    R_S = My_R_sk_Gate(theta_sk, dagger=True, correction_factor=X_SK_Test[1])
    w = R_S.on(cirq.LineQubit(1))
    print(cirq.Circuit.from_ops(w))


class R_sk_DAGGER(cirq.Gate):

    def __init__(self, X_sk, theta_sk, dagger=True):
        """""
    Circuit to build a R_sk^DAGGER gate ... eq (12) arXiv: 1908.08067

    :param theta_sk: Value of theta_sk angle.
    :type theta_sk: float


    :param X_sk: A tuple of tuples: (PauliWord P, constant).
    :type X_sk: tuple
        e.g. ('X0 X1 X2 Y3', -0.28527408634774526j)

    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: A circuit object to be used by cirq.Circuit.from_ops
    :rtype: class

        """
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
    def __init__(self, X_sk):
        """
         blah

        ...
        :raises [ErrorType]: [ErrorDescription]
        ...
        :return: A circuit object to be used by cirq.Circuit.from_ops
        :rtype: class
       """
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
    def __init__(self, X_sk):
        """
         Circuit to perform change of basis in order to perform: e^(-i theta_sk/2 X_sk) ... eq (12) arXiv: 1908.08067

        :param X_sk: A tuple of tuples: ((PauliWord P_s, constant), (PauliWord P_k)). note have P_s and P_k.
        :type X_sk: tuple

        e.g.: X_sk =
        e.g. ('X0 X1 X2 Y3', -0.28527408634774526j)

        Then build circuit!

        ...
        :raises [ErrorType]: [ErrorDescription]
        ...
        :return: A circuit object to be used by cirq.Circuit.from_ops
        :rtype: class
       """
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
    def __init__(self, X_sk, theta_sk, dagger):
        """
        :param X_sk: A tuple of tuples: (PauliWord P, constant).
        :type X_sk: tuple
        e.g. ('X0 X1 X2 Y3', -0.28527408634774526j)

        :param theta_sk: angle
        :type theta_sk: complex float


        ...
        :raises [ErrorType]: [ErrorDescription]
        ...
        :return: A circuit object to be used by cirq.Circuit.from_ops
        :rtype: class
       """
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
    Function takes in dictionary with X_sk and theta_sk information and outputs quantum circuits with
     correction factor for each R_s_k operator per anti-commuting set (given by each key).
     (input from Get_X_sk_operators function)


    :param X_sk_and_theta_sk: A dictionary of X_sk values, with correction factor!
    :type X_sk_and_theta_sk: dict

        e.g.
            {
            'X_sk_theta_sk':    [     {'X_sk': ('X0 X1 X2 Y3', (0.28527408634774526+0j)),
                                      'theta_sk': (0.34438034648829496+0j)},

                                     {'X_sk': ('Y0 I1 I2 I3', (-0.28527408634774526+0j)),
                                        'theta_sk': (0.3423076794345934+0j)}       ],

             'PauliWord_S': ('Z0 I1 I2 I3', (1+0j)),
             'gamma_l': (0.1538026463340925+0j)
             }


    :return: dictionary of R_sk circuits with corresponding correction factor
    :rtype: dict
    e.g.
        [
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
    def __init__(self, anti_commuting_sets, full_anstaz_circuit, S=0):
        self.anti_commuting_sets = anti_commuting_sets
        self.S = S
        super().__init__(anti_commuting_sets, S=S)
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
                iX_sk_constant=1

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
    zz = UnitaryPartition(anti_commuting_sets, ANSATZ, S=0)
    zz.Get_Quantum_circuits_and_constants()
    zz.circuits_and_constants

