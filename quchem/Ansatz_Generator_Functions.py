import numpy as np
import cirq

def HF_state_generator(n_electrons, n_qubits):
    """
     Generate ground state HF state (singlet) in occupation number basis (canonical orbitals)

    Args:
        n_electrons (int): Number of electrons
        n_qubits (int): Number of qubits

    Returns:
        np.array: HF singlet ground state in occupation number basis

.. code-block:: python
   :emphasize-lines: 2

   from quchem.Ansatz_Generator_Functions import *
   state = HF_state_generator(2, 4)
   print(state)
   >> [0. 0. 1. 1.]
    """
    occupied = np.ones(n_electrons)
    unoccupied = np.zeros(n_qubits-n_electrons)
    return np.array([*unoccupied,*occupied])



def Get_Occupied_and_Unoccupied_sites(HF_State):
    """
     Input is HF state in occupation number basis (canonical orbitals)
    e.g. |0011>  =  [0,0,1,1]
    Returns 4 lists of:
    1. spin up sites occupied
    2. spin down sites occupied
    3. spin up sites unoccupied
    4. spin down sites unoccupied

    Args:
        HF_State (list): A list description of HF state... note that indexing from far right to left.

    Raises:
        ValueError: HF state not in occupation number basis

    Returns:
        list: Returns 4 lists of spin up sites occupied, spin down sites occupied, spin up sites unoccupied and finally
              spin down sites unoccupied

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

    notes:
    https://iopscience.iop.org/article/10.1088/2058-9565/aad3e4/pdf

    Args:
        up_occ (list): sites that are spin UP and occupied
        down_occ (list): sites that are spin DOWN and occupied
        up_unocc (list): sites that are spin UP and UN-occupied
        down_unocc (list): sites that are spin down and UN-occupied
        const (float): Constant factor to times operator


    Returns:
        np.array: Two lists of ia and ijab terms

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

    Args:
        ia_terms (list): sites that are spin UP and occupied

    Returns:
        list: List of T1 Terms, where each object in list is a FermionOperator (openfermion.ops.FermionOperator)

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

    Gives list of T2 terms from defined ijab_terms.

    Args:
        ijab_terms (list): list of ijab terms

    Returns:
        list: List of T2 Terms, where each object in list is a FermionOperator (openfermion.ops.FermionOperator)

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

    Input is list of FermionOperator, returns complex transpose.

    Standard case is: T1 or T2 list, returns T1 dagger or T2 dagger (complex transpose)

    Args:
        T_list (list): list of FermionOperator Operators. (openfermion.ops.FermionOperator)

    Returns:
        list:  List of T dagger Terms (complex transpose), where each object in
               list is a FermionOperator (openfermion.ops.FermionOperator)

    """

    from openfermion.utils import hermitian_conjugated

    dagger_terms_list = []
    for term in T_list:
        dagger_terms_list.append(hermitian_conjugated(term))
    return dagger_terms_list


def JW_transform(T_Terms, T_dagger_terms):
    """

    Input is list of FermionOperators and their corresponding complex transpose. Performs the JW transform and
    returns list of QubitOperators.

    Standard case is: T1 and T1 dagger OR T2 list and T2 dagger

    Args:
        T_Terms (list): list of FermionOperator Operators. (openfermion.ops.FermionOperator)
        T_dagger_terms (list): list of FermionOperator Operators. (openfermion.ops.FermionOperator)

    Returns:
        list:  A list containing Pauli Operators for each term. Note each object in
               list is a QubitOperator (openfermion.QubitOperator)

    """

    from openfermion import jordan_wigner
    T_Term_paulis = []
    for i in range(len(T_Terms)):
        T_Term_paulis.append(jordan_wigner(T_Terms[i] - T_dagger_terms[i]))
    return T_Term_paulis


# def Reformat_Pauli_terms(T_Term_Paulis):
#     """
#      Input is list of T Pauli QubitOperators. Output is list of lists to turn into quantum circuit.
#
#      e.g.
#      input (type = QubitOperator)
#      [
#         -0.125j [X0 Z1 Y2] + 0.125j [Y0 Z1 X2],
#         -0.125j [X1 Z2 Y3] + 0.125j [Y1 Z2 X3]
#      ]
#
#      output (type = list of lists, where inner list is QubitOperator)
#     [
#          [0.125j [Y0 Z1 X2], -0.125j [X0 Z1 Y2]],
#          [0.125j [Y1 Z2 X3], -0.125j [X1 Z2 Y3]]
#     ]
#     :param T_Term_Paulis: A list containing QubitOperator (OpenFermion) for each T term
#     :type T_Term_Paulis: list
#
#
#     ...
#     :raises [ErrorType]: [ErrorDescription]
#     ...
#     :return: A list of lists, where each term in list is QubitOperator (openfermion)
#     :rtype: list
#
#
#     """
#
#     Complete_Operation_list = []
#     for term in T_Term_Paulis:
#         sub_term_list = list(term)
#         QubitOperatorSubList = [sub_term for sub_term in sub_term_list]
#         Complete_Operation_list.append(QubitOperatorSubList)
#     return Complete_Operation_list

def Reformat_Pauli_terms(T_Terms_Paulis):
    """

     Input is list of (T Pauli) QubitOperators. Output is list of lists of PauliWords with factors
     to turn into quantum circuit.


    Args:
        T_Terms_Paulis (list): a list of T1 or T2 Terms, where each object in list is a
                                QubitOperator (openfermion.QubitOperator)

    Returns:
        list:  A list of lists... where inner list contains (PauliWord, factor) of T_Terms_Paulis


    from openfermion.ops._qubit_operator import QubitOperator
    T1_Terms_Paulis = [(- QubitOperator('X0 Z1 Y2', 0.125j) + QubitOperator('Y0 Z1 X2', 0.125j)),
        (- QubitOperator('X1 Z2 Y3', 0.125j) + QubitOperator('Y1 Z2 X3', 0.125j))]

.. code-block:: python
   :emphasize-lines: 10

   from openfermion.ops._qubit_operator import QubitOperator
   T1_Terms_Paulis = [(- QubitOperator('X0 Z1 Y2', 0.125j) + QubitOperator('Y0 Z1 X2', 0.125j)),
    (- QubitOperator('X1 Z2 Y3', 0.125j) + QubitOperator('Y1 Z2 X3', 0.125j))]

    >> [
        -0.125j [X0 Z1 Y2] + 0.125j [Y0 Z1 X2],
        -0.125j [X1 Z2 Y3] + 0.125j [Y1 Z2 X3]
       ]

    Reformat_Pauli_terms(T1_Terms_Paulis)
    >> [
            [('Y0 Z1 X2', 0.125j), ('X0 Z1 Y2', -0.125j)],
            [('Y1 Z2 X3', 0.125j), ('X1 Z2 Y3', -0.125j)]
        ]
    """

    PauliWord_list = []

    def digits(P_String):
        return int(P_String[1::])

    for T_term in T_Terms_Paulis:
        temp_list = []
        for qubitNo_qubitOp, constant in T_term.terms.items():
            PauliStrings = [var[1] + str(var[0]) for var in qubitNo_qubitOp]
            qubitNo_list = [var[0] for var in qubitNo_qubitOp]

            max_qubit = int(PauliStrings[-1][1::])
            Q_list = [i for i in range(max_qubit + 1)]

            not_indexed = [qNo for qNo in Q_list if
                           qNo not in qubitNo_list]

            Identity_terms = ['I{}'.format(kk) for kk in not_indexed]

            # seperator = ' '
            # PauliWord = seperator.join(PauliStrings)
            # missing_I = seperator.join(Identity_terms)

            PauliStrings = [*PauliStrings, *Identity_terms]
            # print(PauliStrings)

            PauliStrings = sorted(PauliStrings, key=lambda x: digits(x))
            seperator = ' '
            PauliWord = seperator.join(PauliStrings)
            temp_list.append((PauliWord, constant))
        PauliWord_list.append(temp_list)        # TODO note reverse order of this temp_list may be SOLUTION to problem... BUT gates putting into correct

    return PauliWord_list



## currently WRONG
# def Get_QubitWise_Commuting_groups(Reformat_T_Pauli_terms):
#     flat_list = [PauliString for sublist in Reformat_T_Pauli_terms for PauliString in sublist]
#
#     Pauli_Word_list = []
#     for PauliString in flat_list:
#         x = [key for key in PauliString.terms]
#         Pauli_Word_list.append(*x)
#     print(Pauli_Word_list)

from quchem.quantum_circuit_functions import State_Prep

class UCC_Terms():
    """

    The UCC_Terms object calculates and retains all the unitary coupled cluster terms.

    Args:
        HF_State (list): A list description of HF state... note that indexing from far right to left.

    Attributes:
        up_occ (list): sites that are spin UP and occupied
        down_occ (list): sites that are spin DOWN and occupied
        up_unocc (list): sites that are spin UP and UN-occupied
        down_unocc (list): sites that are spin down and UN-occupied


        ia_terms (list): sites that are spin UP and occupied
        ijab_terms (list): list of ijab terms

        T1_terms (list): list of T1 Terms (list of FermionOperator Operators [openfermion.ops.FermionOperator])
        T2_terms (list): list of T2 Terms (list of FermionOperator Operators [openfermion.ops.FermionOperator])


        T1_dagger_terms (list): list of T1_Dagger Terms (list of FermionOperator Operators [openfermion.ops.FermionOperator])
        T2_dagger_terms (list): list of T2_Dagger Terms (list of FermionOperator Operators [openfermion.ops.FermionOperator])

        T1_Term_paulis (str): JW transform of T1 and T1_dagger. List containing Pauli Operators for each term.
                               Note each object in list is a QubitOperator (openfermion.QubitOperator)
        T2_Term_paulis (str):  JW transform of T2 and T2_dagger. List containing Pauli Operators for each term.
                               Note each object in list is a QubitOperator (openfermion.QubitOperator)

        T1_formatted (list): List of lists... where inner list contains (PauliWord, factor) of T1 terms
        T2_formatted (list): List of lists... where inner list contains (PauliWord, factor) of T2 terms

        HF_prep_quantum_circuit : This is where the HF quantum circuit is stored, from Get_HF_quantum_circuit method.

    """

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

        self.HF_prep_quantum_circuit = None

    def Get_HF_quantum_circuit(self):
        """
       Given HF state in occupation number basis (canonical orbitals), generate cirq circuit to give state.

        Returns:
            HF_prep_quantum_circuit (list): cirq quantum circuit
        """
        HF_state_prep = State_Prep(self.HF_State)
        HF_state_prep_circuit = cirq.Circuit.from_ops(cirq.decompose_once(
            (HF_state_prep(*cirq.LineQubit.range(HF_state_prep.num_qubits())))))

        self.HF_prep_quantum_circuit = HF_state_prep_circuit.all_operations()


import random
import math

def combined_T1_T2_theta_list(T1_formatted, T2_formatted, T1_and_T2_theta_list=[]):

    """

    Args:
        T1_formatted (list): List of lists... where inner list contains (PauliWord, factor) of T1 terms
        T2_formatted (list): List of lists... where inner list contains (PauliWord, factor) of T2 terms
        T1_and_T2_theta_list (list): List of angles, where T1 defined first... followed by T2 angles.

    Raises:
        ValueError: Not enough angles defined

    Returns:
        T1_theta_list (list): list of T1 angles
        T2_theta_list (list) list of T2 angles

    """

    if T1_and_T2_theta_list== []:
        T1_theta_list = [random.uniform(0, 2*math.pi) for i in range(len(T1_formatted))]
        T2_theta_list = [random.uniform(0, 2*math.pi) for i in range(len(T2_formatted))]
    else:
        length_T1 = len(T1_formatted)
        length_T2 = len(T2_formatted)

        if len(T1_and_T2_theta_list) != length_T1 + length_T2:
            raise ValueError('Not enough angles defined. Have {} instead of {} angles.'
                             'ALTERNATIVELY one can use an empty list to generate random angles.'.format(len(T1_and_T2_theta_list), (length_T1 + length_T2)))

        T1_theta_list = [T1_and_T2_theta_list[i] for i in range(length_T1)]
        T2_theta_list = [T1_and_T2_theta_list[i + length_T1] for i in range(length_T2)]

    return T1_theta_list, T2_theta_list


def Set_circuit_angles(T_Terms_Reformatted_Paulis, theta_list=[]):
    """

    Args:
        T_Terms_Reformatted_Paulis (list): list of PauliWords and constants

        theta_list (list): List of theta angles corresponding to each term in T_term. Note if none given, then a
                           randomly generated sequence of numbers if given.

    Returns:
        List of T term and corresponding angle.

.. code-block:: python
   :emphasize-lines: 18

   from openfermion.ops._qubit_operator import QubitOperator
   import numpy as np

   T1_Terms_Paulis = [(- QubitOperator('X0 Z1 Y2', 0.125j) + QubitOperator('Y0 Z1 X2', 0.125j)),
    (- QubitOperator('X1 Z2 Y3', 0.125j) + QubitOperator('Y1 Z2 X3', 0.125j))]

    >> [
        -0.125j [X0 Z1 Y2] + 0.125j [Y0 Z1 X2],
        -0.125j [X1 Z2 Y3] + 0.125j [Y1 Z2 X3]
       ]

    T1_Reformatted_Paulis = Reformat_Pauli_terms(T1_Terms_Paulis)
    >> [
            [('Y0 Z1 X2', 0.125j), ('X0 Z1 Y2', -0.125j)],
            [('Y1 Z2 X3', 0.125j), ('X1 Z2 Y3', -0.125j)]
        ]

    Set_circuit_angles(T1_Reformatted_Paulis, theta_list=[np.pi, 0.1])
    >> [
            ([('Y0 Z1 X2', 0.125j), ('X0 Z1 Y2', -0.125j)], 3.141592653589793),
            ([('I0 Y1 Z2 X3', 0.125j), ('I0 X1 Z2 Y3', -0.125j)], 0.1)
        ]
    """

    if theta_list == []:
        theta_list = [random.uniform(0, 2*math.pi) for i in range(len(T_Terms_Reformatted_Paulis))]

    return list(zip(T_Terms_Reformatted_Paulis, theta_list))

from quchem.quantum_circuit_functions import *

def Get_T_term_circuits(T_Terms_Reformatted_Paulis_and_ANGLES):
    """

    Args:
        T_Terms_Reformatted_Paulis_and_ANGLES (list): List of T term and corresponding angle.

    Returns:
        T_Term_Ansatz_circuits (list): list of cirq circuit generators.


    from openfermion.ops._qubit_operator import QubitOperator
    T1_Terms_Paulis = [(- QubitOperator('X0 Z1 Y2', 0.125j) + QubitOperator('Y0 Z1 X2', 0.125j)),
                        (- QubitOperator('X1 Z2 Y3', 0.125j) + QubitOperator('Y1 Z2 X3', 0.125j))]


    T_Terms_Reformatted_Paulis_and_ANGLES = Set_circuit_angles(T1_Terms_Paulis)
        # [
        #     ([('Y0 Z1 X2', 0.125j), ('X0 Z1 Y2', -0.125j)], 5.575564289159186),
        #     ([('I0 Y1 Z2 X3', 0.125j), ('I0 X1 Z2 Y3', -0.125j)], 4.075039042485969)
        #  ]

        Get_T_term_circuits(T_Terms_Reformatted_Paulis_and_ANGLES)

        # [
        #     [<quchem.quantum_circuit_functions.full_exponentiated_PauliWord_circuit at 0x7f5921dbfb38>,
        #     <quchem.quantum_circuit_functions.full_exponentiated_PauliWord_circuit at 0x7f5921dbfac8>],
        #     [<quchem.quantum_circuit_functions.full_exponentiated_PauliWord_circuit at 0x7f5921dbfbe0>,
        #     <quchem.quantum_circuit_functions.full_exponentiated_PauliWord_circuit at 0x7f5921dbfb00>]
        # ]
    """


    T_Term_Ansatz_circuits = []

    for i in range(len(T_Terms_Reformatted_Paulis_and_ANGLES)):
        angle = T_Terms_Reformatted_Paulis_and_ANGLES[i][1]
        sub_term_circuits = []
        for PauliWord, const in T_Terms_Reformatted_Paulis_and_ANGLES[i][0][::-1]:
            # note reverse order here due to QM notation going from right to left!
            if const.imag < 0:
                sub_term_circuits.append(full_exponentiated_PauliWord_circuit((PauliWord, const), -1*angle))
            else:
                sub_term_circuits.append(full_exponentiated_PauliWord_circuit((PauliWord, const), angle))
        T_Term_Ansatz_circuits.append(sub_term_circuits)

    return T_Term_Ansatz_circuits

    # for term, angle in T_Terms_Reformatted_Paulis_and_ANGLES:
    #     sub_term_circuits = []
    #     for PauliWord in term:
    #         sub_term_circuits.append(full_exponentiated_PauliWord_circuit(PauliWord, angle))
    #     T_Term_Ansatz_circuits.append(sub_term_circuits)
    # return T_Term_Ansatz_circuits


#
# def Trotter_ordering(T_Terms,
#                      trotter_number=1,
#                      trotter_order=1,
#                      term_ordering=None,
#                      k_exp=1.0):
#
#     """
#     https: // github.com / quantumlib / OpenFermion / blob / master / src / openfermion / utils / _trotter_exp_to_qgates.py
#
#     Trotter-decomposes operators into groups without exponentiating
#
#     :param T_Terms: ... Note list of QubitOperators
#     :type T_Terms:
#
#     :param trotter_number: optional number of trotter steps
#     :type trotter_number: int
#
#     :param trotter_order: optional order of trotterization
#     :type trotter_order:
#
#     :param term_ordering:
#     :type term_ordering:
#
#     :param k_exp:
#     :type k_exp:
#
#     ...
#     :raises [ErrorType]: [ErrorDescription]
#     ...
#     :yield: A list containing Pauli Operators for each term. Note each object in list is a QubitOperator (openfermion)
#     :rtype: list
#
#     Note:
#         The default term_ordering is simply the ordered keys of
#         the QubitOperators.terms dict.
#
#     """
#
#     #TODO
#
#
#
# def Trotterisation(T_Terms,
#                      trotter_number=1,
#                      trotter_order=1):
#
#     """
#     Trotter-decomposes operators into groups without exponentiating
#
#     :param T_Terms: ... Note list of QubitOperators
#     :type T_Terms:
#
#     :param trotter_number: optional number of trotter steps
#     :type trotter_number: int
#
#     :param trotter_order: optional order of trotterization
#     :type trotter_order:
#
#     :param term_ordering:
#     :type term_ordering:
#
#     ...
#     :raises [ErrorType]: [ErrorDescription]
#     ...
#     :return: A list containing Pauli Operators for each term. Note each object in list is a QubitOperator (openfermion)
#     :rtype: list
#
#
#     """
#
#     trotter = []
#     for group in T_Terms:
#         l = [(key, value, trotter_order) for key, value in group.terms.items()]
#         trotter.append(l)
#     return trotter




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

    xx =Set_circuit_angles(UCC.T1_formatted, theta_list=[math.pi, 2*math.pi])
    yy = Get_T_term_circuits(xx)

    for sub_term in yy:
        for circuit in sub_term:
            print(cirq.Circuit.from_ops(cirq.decompose_once(
                (circuit(*cirq.LineQubit.range(circuit.num_qubits()))))))


# print(cirq.Circuit.from_ops(
#     [
#     cirq.decompose_once(T1_Ansatz_circuits[0][0](*cirq.LineQubit.range(T1_Ansatz_circuits[0][0].num_qubits()))),
#     cirq.decompose_once(
#             T1_Ansatz_circuits[0][1](*cirq.LineQubit.range(T1_Ansatz_circuits[0][1].num_qubits())))
#     ]
#             ))

class Full_state_prep_circuit(UCC_Terms):
    """

    The Full_state_prep_circuit object calculates and retains all the things required to implement
    unitary coupled cluster single double.

    Args:
        HF_State (list): A list description of HF state... note that indexing from far right to left.

    Attributes:
        T1_theta_list (list): list of T1 angles
        T2_theta_list (list) list of T2 angles

        T1_PauliWords_and_circuits_ANGLES (list): list of PauliWords and constants
        T1_Ansatz_circuits (list): list of cirq circuit generators.

        T2_PauliWords_and_circuits_ANGLES (list): list of PauliWords and constants
        T2_Ansatz_circuits (list): list of cirq circuit generators.


        T1_full_circuit (list): This is where the T1 quantum circuit is stored, from Combine_T1_circuits method.
        T2_full_circuit (list): This is where the T2 quantum circuit is stored, from Combine_T2_circuits method.

        UCC_full_circuit (list): his is where the UCC quantum circuit is stored, from complete_UCC_circuit method.
    """


    def __init__(self, HF_State,  T1_and_T2_theta_list=[]):
        super().__init__(HF_State)

        if self.HF_prep_quantum_circuit ==None:
            self.Get_HF_quantum_circuit()

        self.theta_T1_list, self.theta_T2_list = combined_T1_T2_theta_list(self.T1_formatted, self.T2_formatted, T1_and_T2_theta_list=T1_and_T2_theta_list)

        self.T1_PauliWords_and_circuits_ANGLES = Set_circuit_angles(self.T1_formatted,
                                                       theta_list=self.theta_T1_list)

        self.T1_Ansatz_circuits = Get_T_term_circuits(self.T1_PauliWords_and_circuits_ANGLES)


        self.T2_PauliWords_and_circuits_ANGLES = Set_circuit_angles(self.T2_formatted,
                                                       theta_list= self.theta_T2_list)

        self.T2_Ansatz_circuits = Get_T_term_circuits(self.T2_PauliWords_and_circuits_ANGLES)

        self.T1_full_circuit = None
        self.T2_full_circuit = None

        self.UCC_full_circuit = None

    def Combine_T1_circuits(self):

        T1_full_circuit = []
        for sub_term in self.T1_Ansatz_circuits: #T1_PauliWords_and_circuits_ANGLES
            for circuit in sub_term:
                T1_full_circuit.append(cirq.decompose_once(
                    circuit(*cirq.LineQubit.range(circuit.num_qubits()))))

        self.T1_full_circuit = T1_full_circuit

    def Combine_T2_circuits(self):
        T2_full_circuit = []
        for sub_term in self.T2_Ansatz_circuits:
            for circuit in sub_term:
                T2_full_circuit.append(cirq.decompose_once(
                    circuit(*cirq.LineQubit.range(circuit.num_qubits()))))

        self.T2_full_circuit = T2_full_circuit

    def complete_UCC_circuit(self):

        if self.T1_full_circuit == None:
            self.Combine_T1_circuits()
        if self.T2_full_circuit == None:
            self.Combine_T2_circuits()

        full_circuit = cirq.Circuit.from_ops(
            [
                *self.HF_prep_quantum_circuit,
                *self.T1_full_circuit,
                *self.T2_full_circuit
            ]
        )
        self.UCC_full_circuit = full_circuit

if __name__ == '__main__':
    HF_initial_state = [0, 0, 1, 1]
    UCC = Full_state_prep_circuit(HF_initial_state, T1_and_T2_theta_list=[0,np.pi,0.5*np.pi])
    UCC.complete_UCC_circuit()
    print(UCC.UCC_full_circuit)