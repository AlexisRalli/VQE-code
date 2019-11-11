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
    obey eq (10) in ArXiv:1908.08067 (sum_j B_j^2 = 1

    Output is a new dictionary, with PauliWords and new constant terms... in other part is correction factor!

    :param anti_commuting_sets: A dictionary of anti-commuting sets, each term inside is a tuple of (PauliWord, Constant)
    :type anti_commuting_sets: dict
    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: New dictionary of anti_commuting_sets, with new constants in Tuples.
    :rtype: dict

    """
    for key, value in anti_commuting_sets.items():
        factor = sum([constant**2 for PauliWord, constant in value])

        terms = []
        for PauliWord, constant in value:
            new_constant = constant/np.sqrt(factor)
            terms.append((PauliWord, new_constant))

        # anti_commuting_sets[key] = [terms, ('factor', factor)] # can also have *terms

        anti_commuting_sets[key] = {'PauliWords': terms, 'factor': factor}

    return anti_commuting_sets

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
    # pick S
    for key in normalised_anti_commuting_sets:
        anti_commuting_set = normalised_anti_commuting_sets[key]['PauliWords']

        if len(anti_commuting_set) > 1:


            k_indexes = [index for index in range(len(anti_commuting_set)) if
                       index != S]

            Op_list = []
            for k in k_indexes:

                X_sk_op =(anti_commuting_set[S], anti_commuting_set[k])

                tan_theta_sk = anti_commuting_set[k][1] / (np.sqrt( anti_commuting_set[S][1] + sum([anti_commuting_set[beta_j][1]**2 for beta_j
                                                                                         in np.arange(1,k, 1)]))) #eqn 16

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
        'XZ': (-1j,'X'),

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
        qubitNo = P_s[i][1]

        PauliString_s =  P_s[i][0]
        PauliString_k = P_k[i][0]

        term = PauliString_s + PauliString_k

        new_Pauli = (convert_term[term], qubitNo, new_constant)

        new_Pauli_terms.append(new_Pauli)

    return new_Pauli_terms


#test
if __name__ == '__main__':
    print(convert_X_sk(ww[7][0]['X_sk']))


def Get_R_S_operator(X_sk_and_theta_sk):
    # TODO --> This function is currently pointless (useful to see print statements!
    #          Will probably delete this!
    """
    Function takes in normalised_anti_commuting_sets and gets each R_sk operator according to
    eq (11) in ArXiv:1908.08067.

    Output is a new dictionary, with PauliWords and new constant terms... in other part is correction factor!

    :param X_sk_and_theta_sk: A dictionary of anti-commuting sets.
     Note this is a dictionary of dictionaries where one dict is a tuple of (PauliWord, Constant). The other is
     a dictionary containing the correction to the cofactor.
    :type normalised_anti_commuting_sets: dict
    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: A dictionary containing each R_sk operators for each anti-commuting subset.
    :rtype: dict
    """
    for key in X_sk_and_theta_sk:
        for i in range(len(X_sk_and_theta_sk[key])):
            X_sk = X_sk_and_theta_sk[key][i]['X_sk']
            theta_sk = X_sk_and_theta_sk[key][i]['theta_sk']
            factor = X_sk_and_theta_sk[key][i]['factor']
            #print(X_sk, theta_sk, factor) # TODO build Q circuit from this info!
            print('X_sk: ', X_sk)
            print('theta_sk: ', theta_sk)
            print('factor: ', factor)

Get_R_S_operator(ww)


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

    #X_SK_Test = (('I0 I1 I2 I3', (0.8918294488900189+0j)), ('Y0 I1 I2 Y3', (0.3198751585326103+0j))   )

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

    #X_SK_Test = (('I0 I1 X2 I3', (0.8918294488900189+0j)), ('Y0 I1 I2 I3', (0.3198751585326103+0j))   )

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

    #X_SK_Test = (('I0 I1 I2 I3', (0.8918294488900189+0j)), ('Y0 I1 I2 Y3', (0.3198751585326103+0j))   )

    Ent_final = Engtangle_final(X_SK_Test)

    print(cirq.Circuit.from_ops((Ent_final(*cirq.LineQubit.range(Ent_final.num_qubits())))))
    print(
        cirq.Circuit.from_ops(cirq.decompose_once((Ent_final(*cirq.LineQubit.range(Ent_final.num_qubits()))))))


