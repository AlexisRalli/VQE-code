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

# for key, value in anti_commuting_sets.items():
#     for PauliWord, constant in value:
#         print(PauliWord)


def Get_beta_j_cofactors(anti_commuting_sets):
    """
    Function takes in anti_commuting_sets and returns anti-commuting sets, but with new coefcators that
    obey eq (10) in ArXiv:1908.08067 (sum_j B_j^2 = 1)

    Output is a new dictionary, with PauliWords and new constant terms... in other part is correction factor!

    :param anti_commuting_sets: A dictionary of anti-commuting sets, each term inside is a tuple of (PauliWord, Constant)
    :type anti_commuting_sets: dict
    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: New dictionary of anti_commuting_sets, with new constants in Tuples.
    :rtype: dict

    """
    new_anti_commuting_sets = {}
    for key, value in anti_commuting_sets.items():
        factor = sum([constant**2 for PauliWord, constant in value])

        terms = []
        for PauliWord, constant in value:
            new_constant = constant/np.sqrt(factor)
            terms.append((PauliWord, new_constant))

        # anti_commuting_sets[key] = [terms, ('factor', factor)] # can also have *terms

        new_anti_commuting_sets[key] = {'PauliWords': terms, 'factor': factor}
    # TODO (can make more efficient by missing terms with only 1 PauliWord)
    # do the following:
    #         if len(value)>1:
    #
    #             factor = sum([constant**2 for PauliWord, constant in value])
    #
    #             terms = []
    #             for PauliWord, constant in value:
    #                 new_constant = constant/np.sqrt(factor)
    #                 terms.append((PauliWord, new_constant))
    #
    #             # anti_commuting_sets[key] = [terms, ('factor', factor)] # can also have *terms
    #
    #             new_anti_commuting_sets[key] = {'PauliWords': terms, 'factor': factor}
    #         else:
    #             for PauliWord, constant in value:
    #                 new_anti_commuting_sets[key] = {'PauliWords': PauliWord, 'factor': constant}
    return new_anti_commuting_sets

#test
if __name__ == '__main__':
    ll = Get_beta_j_cofactors(anti_commuting_sets)
    print(ll[10]['PauliWords'])
    print(ll[10]['factor'])


def Get_X_sk_operators(normalised_anti_commuting_sets, S=0): # TODO write function to select 'best' S term!
    """
    Function takes in normalised_anti_commuting_sets and gets each R_sk operator according to
    eq (11) in ArXiv:1908.08067.

    Output is a new dictionary, with PauliWords and new constant terms... in other part is correction factor!
    NOTE S is fixed here!!!

    :param normalised_anti_commuting_sets: A dictionary of anti-commuting sets.
     Note this is a dictionary of dictionaries where one dict is a tuple of (PauliWord, Constant). The other is
     a dictionary containing the correction to the cofactor.
    :type normalised_anti_commuting_sets: dict

    :param S: Index of s in R_sk operator. Note that default is zero. TODO can automate this choice!
    :type S: int
    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: A dictionary containing each X_sk operators for each anti-commuting subset, with associated
    theta_sk value and constant correction factor.

    NOTE: each element of the outermost dict is a list of sub dictionaries - each associated to one sk term.

    :rtype: dict
    """
    X_sk_and_theta_sk={}

    for key in normalised_anti_commuting_sets:
        anti_commuting_set = normalised_anti_commuting_sets[key]['PauliWords']

        if len(anti_commuting_set) > 1:


            k_indexes = [index for index in range(len(anti_commuting_set)) if
                       index != S]

            Op_list = []
            beta_S = None
            for k in k_indexes:

                X_sk_op =(anti_commuting_set[S], anti_commuting_set[k])

                if beta_S == None:
                    beta_S = anti_commuting_set[S][1]
                    tan_theta_sk = anti_commuting_set[k][1] / beta_S # B_k/B_s
                else:
                    tan_theta_sk = anti_commuting_set[k][1] / beta_S # B_k/B_s

                beta_S = np.sqrt(beta_S**2 + anti_commuting_set[k][1]**2) # beta_s_new = (B_s^2 + B_k^2)^0.5 ArXiv:1908.08067

                theta_sk = np.arctan(tan_theta_sk)

                #Op_list.append((X_sk_op, tan_theta_sk, normalised_anti_commuting_sets[key]['factor']))

                Op_list.append({'X_sk': X_sk_op, 'theta_sk': theta_sk, 'factor': normalised_anti_commuting_sets[key]['factor']})

            X_sk_and_theta_sk.update({key: Op_list})

    return X_sk_and_theta_sk

#test
if __name__ == '__main__':
    ww = Get_X_sk_operators(ll, S=0)

    print(ww[7][0]['X_sk'])
    print(ww[7][0]['theta_sk'])
    print(ww[7][0]['factor'])

    print(ww[7][1]['X_sk'])
    print(ww[7][1]['theta_sk'])
    print(ww[7][1]['factor'])


## now have everything for eqn 15 and thus eqn 17!
# 1. apply R_S gate
# 2. Results in P_s being left over :)
# 3. Maybe build this in Cirq!

# do rest in cirq!

def convert_X_sk(X_sk):
    """
    converts i P_s P_k of X_sk into one term!
    (Note that P_k is performed first followed by P_s)

    :param X_sk:

    e.g. : (
              ('Z0 I1 I2 I3', (0.8918294488900189+0j)),
              ('Y0 X1 X2 Y3', (0.3198751585326103+0j))
            )

    :return: Returns a list of tuples containing  (new PauliString, qubitNo, new constant)
    :rtype: list

    e.g. [
            (((-0-1j), 'X'),      '0',     (0.28527408634774526+0j)),
            ((1, 'X'),            '1',     (0.28527408634774526+0j)),
            ((1, 'X'),            '2',     (0.28527408634774526+0j)),
            ((1, 'Y'),            '3',     (0.28527408634774526+0j))
         ]
    (cofactor, New_Pauli_string)  #QubitNo    #new constant
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

    new_constant = X_sk[0][1] * X_sk[1][1]

    P_s = X_sk[0][0].split(' ')
    P_k = X_sk[1][0].split(' ')

    new_Pauli_terms = []
    for i in range(len(P_s)):
        qubitNo = P_s[i][1::]

        PauliString_s =  P_s[i][0]
        PauliString_k = P_k[i][0]

        term = PauliString_s + PauliString_k

        try:
            new_Pauli = (convert_term[term], qubitNo, new_constant)
            new_Pauli_terms.append(new_Pauli)
        except:
            raise KeyError('Cannot combine: {}, as contains Non-Pauli operators'.format(term))


    return new_Pauli_terms


#test
if __name__ == '__main__':
    print(convert_X_sk(ww[7][0]['X_sk']))


# def Get_R_S_operator(X_sk_and_theta_sk):
#     # TODO --> This function is currently pointless (useful to see print statements!
#     #          Will probably delete this!
#     """
#     Function takes in normalised_anti_commuting_sets and gets each R_sk operator according to
#     eq (11) in ArXiv:1908.08067.
#
#     Output is a new dictionary, with PauliWords and new constant terms... in other part is correction factor!
#
#     :param X_sk_and_theta_sk: A dictionary of anti-commuting sets.
#      Note this is a dictionary of dictionaries where one dict is a tuple of (PauliWord, Constant). The other is
#      a dictionary containing the correction to the cofactor.
#     :type normalised_anti_commuting_sets: dict
#     ...
#     :raises [ErrorType]: [ErrorDescription]
#     ...
#     :return: A dictionary containing each R_sk operators for each anti-commuting subset.
#     :rtype: dict
#     """
#     for key in X_sk_and_theta_sk:
#         for i in range(len(X_sk_and_theta_sk[key])):
#             X_sk = X_sk_and_theta_sk[key][i]['X_sk']
#             theta_sk = X_sk_and_theta_sk[key][i]['theta_sk']
#             factor = X_sk_and_theta_sk[key][i]['factor']
#             #print(X_sk, theta_sk, factor) # TODO build Q circuit from this info!
#             print('X_sk: ', X_sk)
#             print('theta_sk: ', theta_sk)
#             print('factor: ', factor)
#
# Get_R_S_operator(ww)

import cirq

class Change_of_Basis_initial(cirq.Gate):
    def __init__(self, X_sk):
        """
         Circuit to perform change of basis in order to perform: e^(-i theta_sk/2 X_sk) ... eq (12) arXiv: 1908.08067

        :param X_sk: A tuple of tuples: ((PauliWord P_s, constant), (PauliWord P_k)). note have P_s and P_k.
        :type X_sk: tuple

        e.g.: X_sk =
                    (
                        ('Z0 I1 I2 I3', (0.8918294488900189+0j)),
                        ('Y0 X1 X2 Y3', (0.3198751585326103+0j))
                    )

        Run convert_X_sk function... giving:
        e.g. [
                (((-0-1j), 'X'),      '0',     (0.28527408634774526+0j)),
                ((1, 'X'),            '1',     (0.28527408634774526+0j)),
                ((1, 'X'),            '2',     (0.28527408634774526+0j)),
                ((1, 'Y'),            '3',     (0.28527408634774526+0j))
             ]
        (cofactor, New_Pauli_string)  #QubitNo    #new constant

        Then build circuit!

        ...
        :raises [ErrorType]: [ErrorDescription]
        ...
        :return: A circuit object to be used by cirq.Circuit.from_ops
        :rtype: class
       """
        self.X_sk = X_sk

        self.X_sk_converted_to_PauliWord = convert_X_sk(self.X_sk)

    def _decompose_(self, qubits):

        for PauliString in self.X_sk_converted_to_PauliWord:

            Pauli_factor, qubitOp = PauliString[0]
            qubitNo =  int(PauliString[1])          # <-- NEED TO MAKE THIS AN INTEGER!
            #new_constant =  PauliString[2]
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
        for i in range(len(self.X_sk_converted_to_PauliWord)):
                Ansatz_basis_change_list.append('Basis_change')
        return Ansatz_basis_change_list

    def num_qubits(self):
        return len(self.X_sk_converted_to_PauliWord)

if __name__ == '__main__':
    X_SK_Test = ww[7][0]['X_sk'] # (  ('Z0 I1 I2 I3', (0.8918294488900189+0j)), ('Y0 X1 X2 Y3', (0.3198751585326103+0j))   )

    #X_SK_Test = (  ('Z0 I1 I2 I3 I4 I5 I6 I7 I8 I9 X10', (0.8918294488900189+0j)), ('Y0 X1 X2 Y3 I4 I5 I6 I7 I8 I9 Z10', (0.3198751585326103+0j))   )

    Basis_change_circuit = Change_of_Basis_initial(X_SK_Test)

    print(cirq.Circuit.from_ops((Basis_change_circuit(*cirq.LineQubit.range(Basis_change_circuit.num_qubits())))))
    print(
        cirq.Circuit.from_ops(cirq.decompose_once((Basis_change_circuit(*cirq.LineQubit.range(Basis_change_circuit.num_qubits()))))))



class Engtangle_initial(cirq.Gate):
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
        self.X_sk_converted_to_PauliWord = convert_X_sk(self.X_sk)

    def _decompose_(self, qubits):

        # note identity  terms removed here
        qubitNo_qubitOp_list = [(int(self.X_sk_converted_to_PauliWord[k][1]), self.X_sk_converted_to_PauliWord[k][0][1])
                                for k in range(len(self.X_sk_converted_to_PauliWord)) if self.X_sk_converted_to_PauliWord[k][0][1] != 'I']

        control_qubit = max([qubitNo for qubitNo, qubitOp in qubitNo_qubitOp_list])


        for qubitNo, qubitOp in qubitNo_qubitOp_list:

            if qubitNo < control_qubit:
                qubitNo_NEXT = qubitNo_qubitOp_list[qubitNo + 1][0]
                yield cirq.CNOT(qubits[qubitNo], qubits[qubitNo_NEXT])


    def _circuit_diagram_info_(self, args):

        qubitNo_qubitOp_list = [(int(self.X_sk_converted_to_PauliWord[k][1]), self.X_sk_converted_to_PauliWord[k][0][1]) for k in range(len(self.X_sk_converted_to_PauliWord))]
        string_list = []
        for i in range(len(qubitNo_qubitOp_list)):
                string_list.append('Entangling circuit')
        return string_list


    def num_qubits(self):
        return len(self.X_sk_converted_to_PauliWord)


if __name__ == '__main__':
    X_SK_Test = ww[7][0]['X_sk'] # (  ('Z0 I1 I2 I3', (0.8918294488900189+0j)), ('Y0 X1 X2 Y3', (0.3198751585326103+0j))   )

    #X_SK_Test = (  ('Z0 I1 I2 I3 I4 I5 I6 I7 I8 I9 X10', (0.8918294488900189+0j)), ('Y0 X1 X2 Y3 I4 I5 I6 I7 I8 I9 Z10', (0.3198751585326103+0j))   )

    Ent_initial = Engtangle_initial(X_SK_Test)

    print(cirq.Circuit.from_ops((Ent_initial(*cirq.LineQubit.range(Ent_initial.num_qubits())))))
    print(
        cirq.Circuit.from_ops(cirq.decompose_once((Ent_initial(*cirq.LineQubit.range(Ent_initial.num_qubits()))))))



class R_sk_DAGGER(cirq.Gate):

    def __init__(self, X_sk, theta_sk):
        """""
    Circuit to build a R_sk^DAGGER gate ... eq (12) arXiv: 1908.08067

    :param theta_sk: Value of theta_sk angle.
    :type theta_sk: float


    :param X_sk: A tuple of tuples: ((PauliWord P_s, constant), (PauliWord P_k)). note have P_s and P_k.
    :type X_sk: tuple

    e.g.: X_sk =
                (
                    ('Z0 I1 I2 I3', (0.8918294488900189+0j)),
                    ('Y0 X1 X2 Y3', (0.3198751585326103+0j))
                )

    Run convert_X_sk function... giving:
    e.g. [
            (((-0-1j), 'X'),      '0',     (0.28527408634774526+0j)),
            ((1, 'X'),            '1',     (0.28527408634774526+0j)),
            ((1, 'X'),            '2',     (0.28527408634774526+0j)),
            ((1, 'Y'),            '3',     (0.28527408634774526+0j))
         ]
    (cofactor, New_Pauli_string)  #QubitNo    #new constant

    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: A circuit object to be used by cirq.Circuit.from_ops
    :rtype: class

        """
        self.X_sk = X_sk
        self.X_sk_converted_to_PauliWord = convert_X_sk(self.X_sk)

        self.theta_sk = theta_sk


    def _decompose_(self, qubits):

        # note identity  terms removed here
        qubitNo_qubitOp_list = [(int(self.X_sk_converted_to_PauliWord[k][1]), self.X_sk_converted_to_PauliWord[k][0][1])
                                for k in range(len(self.X_sk_converted_to_PauliWord)) if self.X_sk_converted_to_PauliWord[k][0][1] != 'I']

        # qubitNo_qubitOp_PauliFactor_list = [(int(self.X_sk_converted_to_PauliWord[k][1]),
        #                                      self.X_sk_converted_to_PauliWord[k][0][1],
        #                                      self.X_sk_converted_to_PauliWord[k][0][0]) ## <-- TODO look at this PauliFactor... e.g. (((-0-1j), 'X') first part! May use to correct Rz rotation here (or can do post)
        #                                     for k in range(len(self.X_sk_converted_to_PauliWord)) if
        #                                     self.X_sk_converted_to_PauliWord[k][0][1] != 'I']

        control_qubit = max([qubitNo for qubitNo, qubitOp in qubitNo_qubitOp_list])

        yield cirq.Rz(self.theta_sk).on(qubits[control_qubit])

    def num_qubits(self):
        return len(self.X_sk_converted_to_PauliWord)

    def _circuit_diagram_info_(self, args):

        qubitNo_qubitOp_list = [(int(self.X_sk_converted_to_PauliWord[k][1]), self.X_sk_converted_to_PauliWord[k][0][1]) for k in range(len(self.X_sk_converted_to_PauliWord))]
        string_list = []
        for i in range(len(qubitNo_qubitOp_list)):
                string_list.append('R_sk_rotation circuit')
        return string_list


if __name__ == '__main__':
    X_SK_Test = ww[7][0]['X_sk'] # (  ('Z0 I1 I2 I3', (0.8918294488900189+0j)), ('Y0 X1 X2 Y3', (0.3198751585326103+0j))   )
    theta_sk = ww[7][0]['theta_sk']

    #X_SK_Test = (  ('Z0 I1 I2 I3 I4 I5 I6 I7 I8 I9 X10', (0.8918294488900189+0j)), ('Y0 X1 X2 Y3 I4 I5 I6 I7 I8 I9 Z10', (0.3198751585326103+0j))   )


    R_sk_rot_circuit = R_sk_DAGGER(X_SK_Test, theta_sk)

    print(cirq.Circuit.from_ops((R_sk_rot_circuit(*cirq.LineQubit.range(R_sk_rot_circuit.num_qubits())))))
    print(
        cirq.Circuit.from_ops(cirq.decompose_once((R_sk_rot_circuit(*cirq.LineQubit.range(R_sk_rot_circuit.num_qubits()))))))


class Engtangle_final(cirq.Gate):
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
        self.X_sk_converted_to_PauliWord = convert_X_sk(self.X_sk)

    def _decompose_(self, qubits):

        # note identity  terms removed here
        qubitNo_qubitOp_list_REVERSE = [(int(self.X_sk_converted_to_PauliWord[k][1]), self.X_sk_converted_to_PauliWord[k][0][1])
                                for k in range(len(self.X_sk_converted_to_PauliWord)) if self.X_sk_converted_to_PauliWord[k][0][1] != 'I'][::-1]

        control_qubit = max([qubitNo for qubitNo, qubitOp in qubitNo_qubitOp_list_REVERSE])

        for i in range(len(qubitNo_qubitOp_list_REVERSE)):
            qubitNo, qubitOp = qubitNo_qubitOp_list_REVERSE[i]

            if qubitNo < control_qubit and qubitNo >= 0:
                qubitNo_NEXT = qubitNo_qubitOp_list_REVERSE[i - 1][0]   # note negative here
                yield cirq.CNOT(qubits[qubitNo], qubits[qubitNo_NEXT])


    def _circuit_diagram_info_(self, args):

        qubitNo_qubitOp_list = [(int(self.X_sk_converted_to_PauliWord[k][1]), self.X_sk_converted_to_PauliWord[k][0][1]) for k in range(len(self.X_sk_converted_to_PauliWord))]
        string_list = []
        for i in range(len(qubitNo_qubitOp_list)):
                string_list.append('Entangling circuit')
        return string_list


    def num_qubits(self):
        return len(self.X_sk_converted_to_PauliWord)


if __name__ == '__main__':
    X_SK_Test = ww[7][0]['X_sk']

    #X_SK_Test = (  ('Z0 I1 I2 I3 I4 I5 I6 I7 I8 I9 X10', (0.8918294488900189+0j)), ('Y0 X1 X2 Y3 I4 I5 I6 I7 I8 I9 Z10', (0.3198751585326103+0j))   )

    Ent_final = Engtangle_final(X_SK_Test)

    print(cirq.Circuit.from_ops((Ent_final(*cirq.LineQubit.range(Ent_final.num_qubits())))))
    print(
        cirq.Circuit.from_ops(cirq.decompose_once((Ent_final(*cirq.LineQubit.range(Ent_final.num_qubits()))))))


class R_sk_full_circuit(cirq.Gate):
    def __init__(self, X_sk, theta_sk):
        """
         blah

        ...
        :raises [ErrorType]: [ErrorDescription]
        ...
        :return: A circuit object to be used by cirq.Circuit.from_ops
        :rtype: class
       """
        self.X_sk = X_sk
        self.X_sk_converted_to_PauliWord = convert_X_sk(self.X_sk)
        self.theta_sk = theta_sk


    def _decompose_(self, qubits):


        Basis_change_circuit = Change_of_Basis_initial(self.X_sk)
        Ent_initial = Engtangle_initial(self.X_sk)
        R_sk_rot_circuit = R_sk_DAGGER(self.X_sk, self.theta_sk)
        Ent_final = Engtangle_final(self.X_sk)

        basis_change_initial_gen = Basis_change_circuit._decompose_(qubits)
        Ent_initial_gen = Ent_initial._decompose_(qubits)
        R_sk_rot_circuit_gen = R_sk_rot_circuit._decompose_(qubits)
        Ent_final_gen = Ent_final._decompose_(qubits)
        basis_change_final_gen = Basis_change_circuit._decompose_(qubits)

        list_generators = [basis_change_initial_gen, Ent_initial_gen, R_sk_rot_circuit_gen, Ent_final_gen,
                           basis_change_final_gen]
        yield list_generators



    def _circuit_diagram_info_(self, args):

        qubitNo_qubitOp_list = [(int(self.X_sk_converted_to_PauliWord[k][1]), self.X_sk_converted_to_PauliWord[k][0][1]) for k in range(len(self.X_sk_converted_to_PauliWord))]
        string_list = []
        for i in range(len(qubitNo_qubitOp_list)):
                string_list.append('R_sk_circuit')
        return string_list


    def num_qubits(self):
        return len(self.X_sk_converted_to_PauliWord)


if __name__ == '__main__':
    X_SK_Test = ww[7][0][
        'X_sk']  # (  ('Z0 I1 I2 I3', (0.8918294488900189+0j)), ('Y0 X1 X2 Y3', (0.3198751585326103+0j))   )
    theta_sk = ww[7][0]['theta_sk']

    # X_SK_Test = (  ('Z0 I1 I2 I3 I4 I5 I6 I7 I8 I9 X10', (0.8918294488900189+0j)), ('Y0 X1 X2 Y3 I4 I5 I6 I7 I8 I9 Z10', (0.3198751585326103+0j))   )

    R_sk_full = R_sk_full_circuit(X_SK_Test, theta_sk)

    print(cirq.Circuit.from_ops((R_sk_full(*cirq.LineQubit.range(R_sk_full.num_qubits())))))
    print(
        cirq.Circuit.from_ops(
            cirq.decompose_once((R_sk_full(*cirq.LineQubit.range(R_sk_full.num_qubits()))))))



# TODO build R_S operator!


def Get_R_S_operators(X_sk_and_theta_sk):
    """
    Function takes in dictionary with X_sk and theta_sk information and outputs quantum circuits with
     correction factor for each R_s_k operator per anti-commuting set (given by each key).


    :param X_sk_and_theta_sk: A dictionary of X_sk values, with correction factor!
    :type X_sk_and_theta_sk: dict

    e.g.
    {
         7: [{'X_sk': (('Z0 I1 I2 I3', (0.8918294488900189+0j)),
            ('Y0 X1 X2 Y3', (0.3198751585326103+0j))),
           'theta_sk': (0.34438034648829496+0j),
           'factor': (0.023655254019369937+0j)},
            {'X_sk': (('Z0 I1 I2 I3', (0.8918294488900189+0j)),
            ('X0 I1 I2 I3', (0.3198751585326103+0j))),
           'theta_sk': (0.325597719954341+0j),
           'factor': (0.023655254019369937+0j)}],
         8: [{'X_sk': (('I0 Z1 I2 I3', (0.9412848366792171+0j)),
            ('Y0 Y1 X2 X3', (-0.33761347164735517+0j))),
           'theta_sk': (-0.344380346488295+0j),
           'factor': (0.021234845659348932+0j)}],
         9: [{'X_sk': (('I0 I1 Z2 I3', (-0.9355920202531878+0j)),
            ('X0 X1 Y2 Y3', (-0.3530829529141257+0j))),
           'theta_sk': (0.36086425264176164-0j),
           'factor': (0.0194148993856907+0j)}],
         10: [{'X_sk': (('I0 I1 I2 Z3', (-0.9355920202531878+0j)),
            ('X0 Y1 Y2 X3', (0.3530829529141257+0j))),
           'theta_sk': (-0.36086425264176164-0j),
           'factor': (0.0194148993856907+0j)}]}

    :return: dictionary of R_sk circuits with corresponding correction factor
    :rtype: dict
    e.g.
    {
         7: [(<__main__.R_sk_full_circuit at 0x7f11771468d0>,  (0.023655254019369937+0j)),
             (<__main__.R_sk_full_circuit at 0x7f1177146a90>,  (0.023655254019369937+0j))],
         8: [(<__main__.R_sk_full_circuit at 0x7f1177146b38>,  (0.021234845659348932+0j))],
         9: [(<__main__.R_sk_full_circuit at 0x7f1177146b70>,  (0.0194148993856907+0j))],
         10: [(<__main__.R_sk_full_circuit at 0x7f1177146ba8>, (0.0194148993856907+0j))]
    }
    """
    output_circuits={}
    for key in X_sk_and_theta_sk:
        list_generators = []
        for terms in X_sk_and_theta_sk[key]:
            R_s_k_circuit_instance = R_sk_full_circuit(terms['X_sk'], terms['theta_sk'])

            correction_factor = terms['factor']

            list_generators.append((R_s_k_circuit_instance, correction_factor))
            # print(
            #     cirq.Circuit.from_ops(
            #         cirq.decompose_once((R_s_k_circuit_instance(*cirq.LineQubit.range(R_s_k_circuit_instance.num_qubits()))))))
            # print(
            #     cirq.Circuit.from_ops(
            #        (R_s_k_circuit_instance(*cirq.LineQubit.range(R_s_k_circuit_instance.num_qubits())))))
        output_circuits[key] = list_generators
    return output_circuits


if __name__ == '__main__':
    X_sk_and_theta_sk = Get_X_sk_operators(ll, S=0)
    bb = Get_R_S_operators(X_sk_and_theta_sk)
    print(cirq.Circuit.from_ops(
       (bb[7][0][0](*cirq.LineQubit.range(bb[7][0][0].num_qubits())))))



#
# class R_S_operator():
#     def __init__(self, X_sk_and_theta_sk_instances):
#         """
#
#         Takes in X_sk_and_theta_sk terms of a  SINGLE anti-commuting set and returns full R_s operator
#         eq (15) ArXiv: 1908.08067
#
#
#         :param X_sk_and_theta_sk_instance: A list of dictionaries containing
#                                            each X_sk term and corresponding theta_sk with the correction factor
#         :type X_sk_and_theta_sk_instances: list
#
#
#         [{'X_sk': (('Z0 I1 I2 I3', (0.8918294488900189+0j)),
#                 ('Y0 X1 X2 Y3', (0.3198751585326103+0j))),
#                'theta_sk': (0.34438034648829496+0j),
#                'factor': (0.023655254019369937+0j)},
#               {'X_sk': (('Z0 I1 I2 I3', (0.8918294488900189+0j)),
#                 ('X0 I1 I2 I3', (0.3198751585326103+0j))),
#                'theta_sk': (0.325597719954341+0j),
#                'factor': (0.023655254019369937+0j)}]
#         """
#         self.X_sk_and_theta_sk_instances = X_sk_and_theta_sk_instances
#
#     def _decompose_(self, qubits):
#
#         list_generators = []
#         for instance in self.X_sk_and_theta_sk_instances:
#             R_s_k_circuit_instance = R_sk_full_circuit(instance['X_sk'], instance['theta_sk'])
#
#             R_s_k_circuit_instance_gen = R_s_k_circuit_instance._decompose_(qubits)
#             list_generators.append(R_s_k_circuit_instance_gen)
#         yield list_generators
#
#         # R_S_circuits_by_key = {}
#         # for key in self.X_sk_and_theta_sk:
#         #     list_generators = []
#         #     for terms in self.X_sk_and_theta_sk[key]:
#         #         R_s_k_circuit_instance = R_sk_full_circuit(terms['X_sk'], terms['theta_sk'])
#         #
#         #         R_s_k_circuit_instance_gen = R_s_k_circuit_instance._decompose_(qubits)
#         #         list_generators.append(R_s_k_circuit_instance_gen)
#         #     R_S_circuits_by_key[key] = list_generators
#         #     yield R_S_circuits_by_key
#
#     def _circuit_diagram_info_(self, args):
#
#         # take anyterm
#         for instance in self.X_sk_and_theta_sk_instances:
#             term = instance['X_sk'][0][0]
#             term = term.split(' ')
#             num_qubits = int(term[-1][1::])
#             break
#
#         string_list = []
#         for i in range(num_qubits + 1):
#                 string_list.append('R_s_circuit (all R_sk sub-circuits!)')
#         return string_list
#
#
#     def num_qubits(self):
#         # take anyterm
#         for instance in self.X_sk_and_theta_sk_instances:
#             term = instance['X_sk'][0][0]
#             term = term.split(' ')
#             num_qubits = int(term[-1][1::])
#             break
#         return num_qubits + 1 # +1 as index starts from 0
#
# if __name__ == '__main__':
#     X_sk_and_theta_sk = Get_X_sk_operators(ll, S=0)
#
#     R_S_full = R_S_operator(X_sk_and_theta_sk[7])
#
#
#     print(cirq.Circuit.from_ops((R_S_full(*cirq.LineQubit.range(R_S_full.num_qubits())))))
#     print(
#         cirq.Circuit.from_ops(
#             cirq.decompose_once((R_S_full(*cirq.LineQubit.range(R_S_full.num_qubits()))))))
#
#
# # class R_S_operator():
# #     def __init__(self, X_sk_and_theta_sk):
# #         """
# #         :param X_sk_and_theta_sk: Dictionary containing each X_sk term and corresponding theta_sk and correction factor
# #         :type X_sk_and_theta_sk: dict
# #
# #         e.g. {7: [{'X_sk': (('Z0 I1 I2 I3', (0.8918294488900189+0j)),
# #                 ('Y0 X1 X2 Y3', (0.3198751585326103+0j))),
# #                'theta_sk': (0.34438034648829496+0j),
# #                'factor': (0.023655254019369937+0j)},
# #               {'X_sk': (('Z0 I1 I2 I3', (0.8918294488900189+0j)),
# #                 ('X0 I1 I2 I3', (0.3198751585326103+0j))),
# #                'theta_sk': (0.325597719954341+0j),
# #                'factor': (0.023655254019369937+0j)}],
# #              8: [{'X_sk': (('I0 Z1 I2 I3', (0.9412848366792171+0j)),
# #                 ('Y0 Y1 X2 X3', (-0.33761347164735517+0j))),
# #                'theta_sk': (-0.344380346488295+0j),
# #                'factor': (0.021234845659348932+0j)}],
# #              9: [{'X_sk': (('I0 I1 Z2 I3', (-0.9355920202531878+0j)),
# #                 ('X0 X1 Y2 Y3', (-0.3530829529141257+0j))),
# #                'theta_sk': (0.36086425264176164-0j),
# #                'factor': (0.0194148993856907+0j)}],
# #              10: [{'X_sk': (('I0 I1 I2 Z3', (-0.9355920202531878+0j)),
# #                 ('X0 Y1 Y2 X3', (0.3530829529141257+0j))),
# #                'theta_sk': (-0.36086425264176164-0j),
# #                'factor': (0.0194148993856907+0j)}]}
# #         """
# #         self.X_sk_and_theta_sk = X_sk_and_theta_sk
# #
# #     def _decompose_(self, qubits):
# #         list_generators=[]
# #         for key in self.X_sk_and_theta_sk:
# #             for terms in self.X_sk_and_theta_sk[key]:
# #                 R_s_k_circuit_instance = R_sk_full_circuit(terms['X_sk'], terms['theta_sk'])
# #
# #                 R_s_k_circuit_instance_gen = R_s_k_circuit_instance._decompose_(qubits)
# #                 list_generators.append(R_s_k_circuit_instance_gen)
# #         return list_generators
# #
# #     def _circuit_diagram_info_(self, args):
# #
# #         # take anyterm
# #         for key in self.X_sk_and_theta_sk:
# #             instance = self.X_sk_and_theta_sk[key][0]['X_sk'][0][0]
# #             instance = instance.split(' ')
# #             num_qubits = int(instance[-1][1::])
# #             break
# #
# #         for i in range(len(num_qubits + 1)):
# #                 string_list.append('R_s_circuit (all R_sk sub-circuits!)')
# #         return string_list
# #
# #
# #     def num_qubits(self):
# #         # take anyterm
# #         for key in self.X_sk_and_theta_sk:
# #             instance = self.X_sk_and_theta_sk[key][0]['X_sk'][0][0]
# #             instance = instance.split(' ')
# #             num_qubits = int(instance[-1][1::])
# #             break
# #         return num_qubits + 1 # +1 as index starts from 0
# #
# #
# # X_sk_and_theta_sk = Get_X_sk_operators(ll, S=0)
# #
# # R_S_full = R_S_operator(X_sk_and_theta_sk)
# #
# # yy = R_S_full._decompose_(cirq.LineQubit.range(4))