import numpy as np

def Get_Occupied_and_Unoccupied_sites(HF_State):
    """
    Input is HF state in occupation number basis (canonical orbitals)
    e.g. |0011>  =  [0,0,1,1]
    Returns 4 lists of:
    1. spin up sites occupied
    2. spin down sites occupied
    3. spin up sites unoccupied
    4. spin down sites unoccupied

    :param HF_State: A list description of HF state... note that indexing from far right to left.
    :type HF_State: list, (numpy.array, tuple)
    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: Returns 4 lists of spin up sites occupied, spin down sites occupied, spin up sites unoccupied and finally
             spin down sites unoccupied
    :rtype: list
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


def Get_ia_and_ijab_terms(up_occ, down_occ, up_unocc, down_unocc, const=0.25):

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

    :param const: Constant factor to times operator
    :type const: float

    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: Two lists of ia and ijab terms
    :rtype: np.array

    notes:
    https://iopscience.iop.org/article/10.1088/2058-9565/aad3e4/pdf

    """

    # SINGLE electron: spin UP transition
    ia_terms = np.zeros((1, 3))
    for i in up_occ:
        for alpha in up_unocc:
            if np.array_equal(ia_terms, np.zeros((1, 3))):
                ia_terms = np.array([alpha, i, const])
                # ia_terms = np.vstack((ia_terms, array))
            else:
                array = np.array([alpha, i, const])
                ia_terms = np.vstack((ia_terms, array))

    # SINGLE electron: spin DOWN transition
    for i in down_occ:
        for alpha in down_unocc:
            if np.array_equal(ia_terms, np.zeros((1, 3))):
                ia_terms = np.array([alpha, i, const])
            else:
                array = np.array([alpha, i, const])
                ia_terms = np.vstack((ia_terms, array))

    ## DOUBLE electron: two spin UP transition
    ijab_terms = np.zeros((1, 5))
    for i in up_occ:
        for j in up_occ:
            if i > j:
                for alpha in up_unocc:
                    for beta in up_unocc:
                        if alpha > beta:
                            if np.array_equal(ijab_terms, np.zeros((1, 5))):
                                ijab_terms = np.array([alpha, beta, i, j, const])
                            else:
                                array = np.array([alpha, beta, i, j, const])
                                ijab_terms = np.vstack((ijab_terms, array))


    ## DOUBLE electron: two spin DOWN transition
    for i in down_occ:
        for j in down_occ:
            if i > j:
                for alpha in down_unocc:
                    for beta in down_unocc:
                        if alpha > beta:
                            if np.array_equal(ijab_terms, np.zeros((1, 5))):
                                ijab_terms = np.array([alpha, beta, i, j, const])
                            else:
                                array = np.array([alpha, beta, i, j, const])
                                ijab_terms = np.vstack((ijab_terms, array))

    ## DOUBLE electron: one spin UP and one spin DOWN transition
    for i in up_occ:
        for j in down_occ:
            if i > j:
                for alpha in up_unocc:
                    for beta in down_unocc:
                        if alpha > beta:
                            if np.array_equal(ijab_terms, np.zeros((1, 5))):
                                ijab_terms = np.array([alpha, beta, i, j, const])
                            else:
                                array = np.array([alpha, beta, i, j, const])
                                ijab_terms = np.vstack((ijab_terms, array))

    ## DOUBLE electron: one spin DOWN and one spin UP transition
    for i in down_occ:
        for j in up_occ:
            if i > j:
                for alpha in down_unocc:
                    for beta in up_unocc:
                        if alpha > beta:
                            if np.array_equal(ijab_terms, np.zeros((1, 5))):
                                ijab_terms = np.array([alpha, beta, i, j, const])
                            else:
                                array = np.array([alpha, beta, i, j, const])
                                ijab_terms = np.vstack((ijab_terms, array))

    # this makes sure we have array of arrays! (not the case if only have one entry... this corrects for this)
    if len(ia_terms.shape) == 1:
        ia_terms = np.array([ia_terms])

    if len(ijab_terms.shape) == 1:
        ijab_terms = np.array([ijab_terms])


    return ia_terms, ijab_terms


def Get_T1_terms_list(ia_terms):
    """
    Gives list of T1 terms from defined ia_terms.

    :param ia_terms: sites that are spin UP and occupied
    :type ia_terms: list, (numpy.array, tuple)

    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: List of T1 Terms, each is a object in list is a FermiOperator (OpenFermion)
    :rtype: list
    """
    from openfermion.ops import FermionOperator

    T1_terms=[]
    for x in range(len(ia_terms)):
        i = int(ia_terms[x][0])
        alph = int(ia_terms[x][1])
        t_ia = float(ia_terms[x][2])
        term = FermionOperator('{}^ {}'.format(i, alph), t_ia)
        T1_terms.append(term)
    return T1_terms


def Get_T2_terms_list(ijab_terms):
    """
    Gives list of T2 terms from defined ia_terms.

    :param ijab_terms: list of ijab
    :type ijab_terms: list, (numpy.array, tuple)

    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: List of T2 Terms, each is a object in list is a FermiOperator (OpenFermion)
    :rtype: list
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

def dagger_T_list(T_list):
    """
     Input T1 or T2 list, returns T1 dagger or T2 dagger (complex transpose)

    :param T_list: list of Fermionic Operators
    :type T_list: list

    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: List of T dagger Terms (complex transpose), each is a object in list is a FermiOperator (OpenFermion)
    :rtype: list

    """
    from openfermion.utils import hermitian_conjugated

    dagger_terms_list = []
    for term in T_list:
        dagger_terms_list.append(hermitian_conjugated(term))
    return dagger_terms_list


def JW_transform(T_Terms, T_dagger_terms):
    """
     Input T1 or T2 list, returns T1 dagger or T2 dagger (complex transpose)

    :param T_Terms: A list containing Fermionic operators for each T term
    :type T_Terms: list

    :param T_dagger_terms: A list containing Fermionic operators for each T term that is complex conjugated. Note order
                           is important
    :type T_dagger_terms: list

    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: A list containing Pauli Operators for each term. Note each object in list is a QubitOperator (openfermion)
    :rtype: list


    """
    from openfermion import jordan_wigner
    T_Term_paulis = []
    for i in range(len(T_Terms)):
        T_Term_paulis.append(jordan_wigner(T_Terms[i] - T_dagger_terms[i]))
    return T_Term_paulis


def Reformat_Pauli_terms(T_Term_Paulis):
    """
     Input is list of T Pauli QubitOperators. Output is list of lists to turn into quantum circuit.

     e.g.
     input (type = QubitOperator)
     [-0.125j [X0 Z1 Y2] + 0.125j [Y0 Z1 X2], -0.125j [X1 Z2 Y3] + 0.125j [Y1 Z2 X3]]

     output (type = list of lists, where inner list is (operation, constant))
     [
        [(((0, 'Y'), (1, 'Z'), (2, 'X')), 0.125j),
        (((0, 'X'), (1, 'Z'), (2, 'Y')), -0.125j)],

        [(((1, 'Y'), (2, 'Z'), (3, 'X')), 0.125j),
        (((1, 'X'), (2, 'Z'), (3, 'Y')), -0.125j)]
    ]

    :param T_Term_Paulis: A list containing QubitOperator (OpenFermion) for each T term
    :type T_Term_Paulis: list


    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: A list containing Pauli Operators for each term.
    :rtype: list


    """
    Complete_Operation_list = []

    for sub_term in T_Term_Paulis:
        sub_instance_list = []
        for qubitNoOp, sign in sub_term.terms.items():
            sub_instance_list.append((qubitNoOp, sign))
        Complete_Operation_list.append(sub_instance_list)

    return Complete_Operation_list

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

        T1_dagger_terms= dagger_T_list(T1_terms)
        self.T1_dagger_terms = T1_dagger_terms

        T2_dagger_terms = dagger_T_list(T2_terms)
        self.T2_dagger_terms = T2_dagger_terms

        self.T1_Term_paulis = JW_transform(T1_terms, T1_dagger_terms)
        self.T2_Term_paulis = JW_transform(T2_terms, T2_dagger_terms)

        self.T1_formatted = Reformat_Pauli_terms(self.T1_Term_paulis)
        self.T2_formatted = Reformat_Pauli_terms(self.T2_Term_paulis)






def Trotter_ordering(T_Terms,
                     trotter_number=1,
                     trotter_order=1,
                     term_ordering=None,
                     k_exp=1.0):

    """
    https: // github.com / quantumlib / OpenFermion / blob / master / src / openfermion / utils / _trotter_exp_to_qgates.py

    Trotter-decomposes operators into groups without exponentiating

    :param T_Terms: ... Note list of QubitOperators
    :type T_Terms:

    :param trotter_number: optional number of trotter steps
    :type trotter_number: int

    :param trotter_order: optional order of trotterization
    :type trotter_order:

    :param term_ordering:
    :type term_ordering:

    :param k_exp:
    :type k_exp:

    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :yield: A list containing Pauli Operators for each term. Note each object in list is a QubitOperator (openfermion)
    :rtype: list

    Note:
        The default term_ordering is simply the ordered keys of
        the QubitOperators.terms dict.

    """

    #TODO



def Trotterisation(T_Terms,
                     trotter_number=1,
                     trotter_order=1):

    """
    Trotter-decomposes operators into groups without exponentiating

    :param T_Terms: ... Note list of QubitOperators
    :type T_Terms:

    :param trotter_number: optional number of trotter steps
    :type trotter_number: int

    :param trotter_order: optional order of trotterization
    :type trotter_order:

    :param term_ordering:
    :type term_ordering:

    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: A list containing Pauli Operators for each term. Note each object in list is a QubitOperator (openfermion)
    :rtype: list


    """

    trotter = []
    for group in T_Terms:
        l = [(key, value, trotter_order) for key, value in group.terms.items()]
        trotter.append(l)
    return trotter




# Test
if __name__ == '__main__':
    #HF_initial_state = [0, 0, 0, 0, 1, 1, 1, 1]
    HF_initial_state = [0, 0, 1, 1]
    UCC = UCC_Terms(HF_initial_state)

    print(UCC.ia_terms)
    print(UCC.ijab_terms)
    print(UCC.T1_Term_paulis)
    print(UCC.T2_Term_paulis)
    print(UCC.T1_Term_paulis[1].terms)
    print(Reformat_Pauli_terms(UCC.T1_Term_paulis))


