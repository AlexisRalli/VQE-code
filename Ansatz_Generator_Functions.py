HF_initial_state = [0,0,0,0,1,1,1,1,1,1,1,1,1,1]

import numpy as np

def Get_Occupied_and_Unoccupied_sites(HF_State):
    """
    Input is HF state in occupation number basis (cannonical orbitals)
    Returns 4 lists of:
    1. spin up sites occupied
    2. spin down sites occupied
    3. spin up sites unoccupied
    4. spin down sites unoccupied

    :param HF_State: [ParamDescription], defaults to [DefaultParamVal]
    :type HF_State: list, (numpy.array, tuple)
    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: [ReturnDescription]
    :rtype: [ReturnType]
    """

    up_occ = []
    down_occ = []
    up_unocc = []
    down_unocc =[]

    for i in range(len(HF_State)):

        bit = HF_State[-1::-1][i]  # note this slice reverses order! (QM indexing)

        if i % 2 == 0 and bit == 1: #modulo two checks if spin up or spin down... bit is if occupied or not
            up_occ.append(i)

        elif bit == 1:
            down_occ.append(i)

        if i % 2 == 0 and bit == 0:
            up_unocc.append(i)

        elif bit == 0:
            down_unocc.append(i)

        if bit != 0 and bit != 1:
            raise ValueError('HF state: {} not in correct format. bit at index {} is /'
                             'not 0 or 1 but {}'.format(HF_State,i, bit))


    return up_occ, down_occ, up_unocc, down_unocc


def Get_ia_and_ijab_terms(up_occ, down_occ, up_unocc, down_unocc):

    """
    Input is lists of occupied and unoccupied sites
    Returns 2 lists of:
    1. ia_terms
    2. ijab terms

    :param up_occ: sites that are spin UP and occupied
    :type up_occ: list, (numpy.array, tuple)

    :param down_occ: sites that are spin DOWN and occupied
    :type down_occ: list, (numpy.array, tuple)

    :param up_unocc: sites that are spin UP and UN-occupied
    :type up_unocc: list, (numpy.array, tuple)

    :param down_unocc: sites that are spin down and UN-occupied
    :type down_unocc: list, (numpy.array, tuple)

    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: Two lists of ia and ijab terms
    :rtype: list
    """

    # SINGLE electron: spin UP transition
    ia_terms = np.zeros((1, 3))
    for i in up_occ:
        for alpha in up_unocc:
            if ia_terms.any() == np.zeros((1, 3)).any():
                ia_terms = np.array([alpha, i, 0.25])
                # ia_terms = np.vstack((ia_terms, array))
            else:
                array = np.array([alpha, i, 0.25])
                ia_terms = np.vstack((ia_terms, array))

    # SINGLE electron: spin DOWN transition
    for i in down_occ:
        for alpha in down_unocc:
            if ia_terms.any() == np.zeros((1, 3)).any():
                ia_terms = np.array([alpha, i, 0.25])
            else:
                array = np.array([alpha, i, 0.25])
                ia_terms = np.vstack((ia_terms, array))

    ## DOUBLE electron: two spin UP transition
    ijab_terms = np.zeros((1, 5))
    for i in up_occ:
        for j in up_occ:
            if i > j:
                for alpha in up_unocc:
                    for beta in up_unocc:
                        if alpha > beta:
                            if ijab_terms.any() == np.zeros((1, 5)).any():
                                ijab_terms = np.array([beta, alpha, j, i, 0.25])
                            else:
                                array = np.array([alpha, beta, i, j, 0.25])
                                ijab_terms = np.vstack((ijab_terms, array))


    ## DOUBLE electron: two spin DOWN transition
    for i in down_occ:
        for j in down_occ:
            if i > j:
                for alpha in down_unocc:
                    for beta in down_unocc:
                        if alpha > beta:
                            if ijab_terms.any() == np.zeros((1, 5)).any():
                                ijab_terms = np.array([beta, alpha, j, i, 0.25])
                            else:
                                array = np.array([alpha, beta, i, j, 0.25])
                                ijab_terms = np.vstack((ijab_terms, array))

    ## DOUBLE electron: one spin UP and one spin DOWN transition
    for i in up_occ:
        for j in down_occ:
            if i > j:
                for alpha in up_unocc:
                    for beta in down_unocc:
                        if alpha > beta:
                            if ijab_terms.any() == np.zeros((1, 5)).any():
                                ijab_terms = np.array([beta, alpha, j, i, 0.25])
                            else:
                                array = np.array([alpha, beta, i, j, 0.25])
                                ijab_terms = np.vstack((ijab_terms, array))

    ## DOUBLE electron: one spin DOWN and one spin UP transition
    for i in down_occ:
        for j in up_occ:
            if i > j:
                for alpha in down_unocc:
                    for beta in up_unocc:
                        if alpha > beta:
                            if ijab_terms.any() == np.zeros((1, 5)).any():
                                ijab_terms = np.array([beta, alpha, j, i, 0.25])
                            else:
                                array = np.array([alpha, beta, i, j, 0.25])
                                ijab_terms = np.vstack((ijab_terms, array))

    return ia_terms, ijab_terms


def Get_T1_terms_list(ia_terms):
    """
    dsf
    """
    from openfermion.ops import FermionOperator

    T1_terms=[]
    for x in range(len(ia_terms)):
        i = int(ia_terms[x][0])
        alph = int(ia_terms[x][1])
        t_ia = float(ia_terms[x][2])

        term = FermionOperator('{}^ {}'.format(i, alph), t_ia)
        # print(term.terms) # correct!

        T1_terms.append(term)
    return T1_terms


def Get_T2_terms_list(ijab_terms):
    """
    ds
    """
    from openfermion.ops import FermionOperator

    T2_terms = []
    for x in range(len(ijab_terms)):
        i = int(ijab_terms[x][0])
        j = int(ijab_terms[x][1])
        alph = int(ijab_terms[x][2])
        beta = int(ijab_terms[x][3])
        t_ijab = float(ijab_terms[x][4])

        term = FermionOperator('{}^ {}^ {} {} '.format(i, j, alph, beta), t_ijab)
        T2_terms.append(term)

    return T2_terms

def daggar_T_list(T_list):
    """
    Args:
        T_list: Lists of Fermionic operators
                Desired input is: T1_terms_list // T2_terms_list funcitons


    Returns:
        A list containing the complex transpose of Fermionic operators
    """
    from openfermion.utils import hermitian_conjugated
    dagger_terms_list = []
    for term in T_list:
        dagger_terms_list.append(hermitian_conjugated(term))
    return dagger_terms_list


def JW_transform(T_Term, T_dagger_terms):
    """
    Arg:
        T_Term = A list containing Fermionic operators for each T term
        T_dagger_terms =  A list containing Fermionic operators for each T dagger term
    Returns:
        A list containing Pauli operators for each T term. Each term in list is a QubitOperator!
    """
    from openfermion import jordan_wigner
    T_Term_pauli = []
    for i in range(len(T_Term)):
        T_Term_pauli.append(jordan_wigner(T_Term[i] - T_dagger_terms[i]))
    return T_Term_pauli




class UCC_Terms():


    def __init__(self, HF_State):

        self.HF_State = HF_State

        up_occ, down_occ, up_unocc, down_unocc = Get_Occupied_and_Unoccupied_sites(HF_State)
        self.up_occ = up_occ
        self.down_occ = down_occ

        self.up_unocc = up_unocc
        self.down_unocc = down_unocc

        ia_terms, ijab_terms = Get_ia_and_ijab_terms(up_occ, down_occ, up_unocc, down_unocc)
        self.ia_terms = ia_terms
        self.ijab_terms = ijab_terms

        T1_terms = Get_T1_terms_list(ia_terms)
        self.T1_terms = T1_terms

        T2_terms = Get_T2_terms_list(ijab_terms)
        self.T2_terms = T2_terms

        T1_dagger_terms= daggar_T_list(T1_terms)
        self.T1_dagger_terms = T1_dagger_terms

        T2_dagger_terms = daggar_T_list(T2_terms)
        self.T2_dagger_terms = T2_dagger_terms

        self.T1_Term_pauli = JW_transform(T1_terms, T1_dagger_terms)
        self.T2_Term_pauli = JW_transform(T2_terms, T2_dagger_terms)



#
# class UCC_Terms():
#
#
#
#     def __init__(self, HF_State):
#         self.HF_State = HF_State
#
#         self.up_occ = []
#         self.down_occ = []
#
#         self.up_unocc = []
#         self.down_unocc = []
#
#         self.ia_terms = None
#         self.ijab_terms = None
#
#         self.T1_terms = []
#         self.T2_terms = []
#
#         self.T1_dagger_terms = []
#         self.T2_dagger_terms = []
#
#         self.T1_Term_pauli = []
#         self.T2_Term_pauli = []
#
#     @property
#     def Get_Occupited(self):
#
#         for i in range(len(self.HF_State)):
#
#             bit = self.HF_State[-1::-1][i] # note this slice reverses order!
#
#             if i%2==0 and bit == 1:
#                 self.up_occ.append(i)
#
#             elif bit == 1:
#                 self.down_occ.append(i)
#
#
#             if i%2==0 and bit == 0:
#                 self.up_unocc.append(i)
#
#             elif bit == 0:
#                 self.down_unocc.append(i)
#
#         return self.up_occ, self.down_occ, self.up_unocc, self.down_unocc
#
#     @property
#     def Get_Terms(self):
#
#
#         ia_terms = np.zeros((1, 3))
#         for i in self.up_occ:
#             for alpha in self.up_unocc:
#                 if ia_terms.any() == np.zeros((1, 3)).any():
#                     ia_terms = np.array([alpha, i, 0.25])
#                     # ia_terms = np.vstack((ia_terms, array))
#                 else:
#                     array = np.array([alpha, i, 0.25])
#                     ia_terms = np.vstack((ia_terms, array))
#
#         ## two spin up
#         ijab_terms = np.zeros((1, 5))
#         for i in self.up_occ:
#             for j in self.up_occ:
#                 if i > j:
#                     for alpha in self.up_unocc:
#                         for beta in self.up_unocc:
#                             if alpha > beta:
#                                 if ijab_terms.any() == np.zeros((1, 5)).any():
#                                     ijab_terms = np.array([beta, alpha, j, i, 0.25])
#                                 else:
#                                     array = np.array([alpha, beta, i, j, 0.25])
#                                     ijab_terms = np.vstack((ijab_terms, array))
#
#         # spin up and spin down
#         for i in self.up_occ:
#             for j in self.down_occ:
#                 if i > j:
#                     for alpha in self.up_unocc:
#                         for beta in self.down_unocc:
#                             if alpha > beta:
#                                 array = np.array([beta, alpha, j, i, 0.25])
#                                 ijab_terms = np.vstack((ijab_terms, array))
#
#         self.ia_terms = ia_terms
#         self.ijab_terms = ijab_terms
#
#     @property
#     def T1_terms_list(self):
#         """
#         Returns a list of T1 terms
#
#         Arg: ia_terms isT1_terms a numpy array.
#             first values are i and p
#             3rd term is the value of t_ia
#             e.g [2,0, 0.25] = exciation from 0 --> 2 with t_ia = 0.25
#         Returns:
#             A list containing Fermionic operators for each T1 term
#         """
#         from openfermion.ops import FermionOperator
#
#
#         for x in range(len(self.ia_terms)):
#             i = int(self.ia_terms[x][0])
#             alph = int(self.ia_terms[x][1])
#             t_ia = float(self.ia_terms[x][2])
#
#             term = FermionOperator('{}^ {}'.format(i, alph), t_ia)
#             # print(term.terms) # correct!
#
#             self.T1_terms.append(term)
#
#     @property
#     def T2_terms_list(self):
#         """
#         Arg:  ijab_terms is T2_terms in a numpy array.
#             first values are i,j = sites to excite too (unoccipied)
#             last values a,b = occupied sites
#             e.g [3,2,1,0,0.5] = exciation from 0,1 --> 2,3 with t_ijab = 0.5
#         Returns:
#             A list containing Fermionic operators for each T2 term
#         """
#         from openfermion.ops import FermionOperator
#
#         for x in range(len(self.ijab_terms)):
#             i = int(self.ijab_terms[x][0])
#             j = int(self.ijab_terms[x][1])
#             alph = int(self.ijab_terms[x][2])
#             beta = int(self.ijab_terms[x][3])
#             t_ijab = float(self.ijab_terms[x][4])
#
#             term = FermionOperator('{}^ {}^ {} {} '.format(i, j, alph, beta), t_ijab)
#             self.T2_terms.append(term)
#
#     def daggar_T_list(self, T_list):
#         """
#         Args:
#             T_list: Lists of Fermionic operators
#                     Desired input is: T1_terms_list // T2_terms_list funcitons
#
#
#         Returns:
#             A list containing the complex transpose of Fermionic operators
#         """
#         from openfermion.utils import hermitian_conjugated
#         dagger_terms_list = []
#         for term in T_list:
#             dagger_terms_list.append(hermitian_conjugated(term))
#         return dagger_terms_list
#
#     def Get_T1_and_T2_dagger(self):
#         self.T1_dagger_terms = self.daggar_T_list(self.T1_terms)
#         self.T2_dagger_terms = self.daggar_T_list(self.T2_terms)
#
#     def JW_transform(self, T_Term, T_dagger_terms):
#         """
#         Arg:
#             T_Term = A list containing Fermionic operators for each T term
#             T_dagger_terms =  A list containing Fermionic operators for each T dagger term
#         Returns:
#             A list containing Pauli operators for each T term. Each term in list is a QubitOperator!
#         """
#         from openfermion import jordan_wigner
#         T_Term_pauli = []
#         for x in range(len(T_Term)):
#             T_Term_pauli.append(jordan_wigner(T_Term[x] - T_dagger_terms[x]))
#         return T_Term_pauli
#
#     def T1_and_T2_JW_transform(self):
#         self.T1_Term_pauli = self.JW_transform(self.T1_terms, self.T1_dagger_terms)
#         self.T2_Term_pauli = self.JW_transform(self.T2_terms, self.T2_dagger_terms)
#
#
#
#
#
# # # MoleculeName = 'H2O'
# up_occ=[2,4,6,8] # spin up
# down_occ=[3,5,7,9] # spin down
#
# up_unocc = [10, 12]
# down_unocc = [11, 13]
#
#
# if __name__ == '__main__':
#     WW = UCC_Terms([0, 0, 0, 0, 1, 1, 1, 1])
#     WW.Get_Occupited()
#     WW.Get_Terms()
#     print(WW.ia_terms)
#     print(WW.ijab_terms)
#     WW.T1_terms_list()
#     print(WW.T1_terms)
#     WW.T2_terms_list()
#     print(WW.T2_terms)
#     WW.Get_T1_and_T2_dagger()
#     print(WW.T1_dagger_terms)
#     print(WW.T1_terms)
#
#     WW.T1_and_T2_JW_transform()
#     print(WW.T1_Term_pauli)
#     print(WW.T2_Term_pauli)
#
#     x = WW.T1_Term_pauli
#     print(x[1])
#     print(x[1].terms)


