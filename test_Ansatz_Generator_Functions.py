from Ansatz_Generator_Functions import *
import pytest
# in Terminal run (LINUX!):
# py.test /home/alexisr/Documents/PhD/Code/PhD\ Code/test_Ansatz_Circuit_Functions.py

def test_Get_Occupied_and_Unoccupied_sites_H2():

    """
    Test to make sure standard use works [H2 example]
    """

    HF_State = [0,0,1,1]

    up_occ_true = [0]  # spin up
    down_occ_true = [1]  # spin down

    up_unocc_true = [2]
    down_unocc_true = [3]

    up_occ, down_occ, up_unocc, down_unocc = Get_Occupied_and_Unoccupied_sites(HF_State)

    assert (up_occ, down_occ, up_unocc, down_unocc) == (up_occ_true, down_occ_true, up_unocc_true, down_unocc_true)

def test_Get_Occupied_and_Unoccupied_sites_H2O():

    """
    Test to make sure standard use works [H2O example]
    """

    HF_State = [0,0,0,0,1,1,1,1,1,1,1,1,1,1]

    up_occ_true = [0,2,4,6,8] # spin up
    down_occ_true = [1,3,5,7,9] # spin down

    up_unocc_true = [10, 12]
    down_unocc_true = [11, 13]

    up_occ, down_occ, up_unocc, down_unocc = Get_Occupied_and_Unoccupied_sites(HF_State)

    assert (up_occ, down_occ, up_unocc, down_unocc) == (up_occ_true, down_occ_true, up_unocc_true, down_unocc_true)

def test_Get_Occupied_and_Unoccupied_sites_DIFFERENT_ORDERING():

    """
    Test to see if function deals with different ordering of states (not only lowest energy levels filled)
    """

    HF_State = [0,0,1,0,1,0,0,1,0,1]

    up_occ_true = [0,2] # spin up
    down_occ_true = [5,7] # spin down

    up_unocc_true = [4,6,8]
    down_unocc_true = [1,3,9]

    up_occ, down_occ, up_unocc, down_unocc = Get_Occupied_and_Unoccupied_sites(HF_State)

    assert (up_occ, down_occ, up_unocc, down_unocc) == (up_occ_true, down_occ_true, up_unocc_true, down_unocc_true)

def test_Get_Occupied_and_Unoccupied_sites_WRONG_NOTATION():
    '''
    Test for when HF state has incorrect notation.
    Occupation basis can only be 0 or 1... testing for larger values
    '''
    HF_State = [0,0,0,1,99,1]


    with pytest.raises(ValueError) as exc_info:

        assert exc_info is Get_Occupied_and_Unoccupied_sites(HF_State)



def test_Get_ia_and_ijab_terms_H2():

    """
    Test to make sure standard use works [H2 example]

    note: H2 HF_state = [0,0,1,1]

    """

    up_occ = [0]
    down_occ = [1]
    up_unocc = [2]
    down_unocc = [3]


    ia_terms_true = np.array([[2, 0,0.25],
                              [3, 1, 0.25]])

    ijab_terms_true = np.array([np.array([3, 2, 1, 0, 0.25])])

    ia_terms, ijab_terms = Get_ia_and_ijab_terms(up_occ, down_occ, up_unocc, down_unocc)

    assert np.array_equal(ia_terms_true, ia_terms) and np.array_equal(ijab_terms_true, ijab_terms)


def test_Get_ia_and_ijab_terms_8_sites():

    # HF_State = [0,0,0,0,1,1,1,1]

    """
    Test to make sure standard use works [8 sites],
    4 Sites occupied and 4 sites unoccupied

    note: HF_state = [0,0,0,0,1,1,1,1]

    """

    up_occ = [0,2]
    down_occ = [1,3]
    up_unocc = [4,6]
    down_unocc = [5,7]


    ia_terms_true = np.array([
                              [4, 0, 0.25],
                              [6, 0, 0.25],

                              [4, 2, 0.25],
                              [6, 2, 0.25],

                              [5, 1, 0.25],
                              [7, 1, 0.25],

                              [5, 3, 0.25],
                              [7, 3, 0.25]
                              ])

    # note ordering in function
    # First is double spin up
    # second is double spin down
    # third is spin up and spin down
    # final is spin down and spin up

    # where i >j and alpha > beta       <-- THIS IS VERY IMPORTANT
    # ordering is : [beta, alpha, j, i]

    ijab_terms_true = np.array([
                                # up-up
                                [6, 4, 2, 0, 0.25],

                                # down-down
                                [7, 5, 3, 1, 0.25],

                                # up-down
                                [6, 5, 2, 1, 0.25],

                                # down-up
                                [5, 4, 1, 0, 0.25],
                                [7, 4, 1, 0, 0.25],
                                [7, 6, 1, 0, 0.25],

                                [5, 4, 3, 0, 0.25],
                                [7, 4, 3, 0, 0.25],
                                [7, 6, 3, 0, 0.25],

                                [5, 4, 3, 2, 0.25],
                                [7, 4, 3, 2, 0.25],
                                [7, 6, 3, 2, 0.25],
    ])



    ia_terms, ijab_terms = Get_ia_and_ijab_terms(up_occ, down_occ, up_unocc, down_unocc)

    assert np.array_equal(ia_terms_true, ia_terms) and np.array_equal(ijab_terms_true, ijab_terms)


def test_Get_T1_terms_list():
    from openfermion.ops import FermionOperator

    const = 0.25

    ia_terms = np.array([[2, 0, const],
                         [3, 1, const]])

    T1_terms = Get_T1_terms_list(ia_terms)

    True_T1_terms = [FermionOperator('2^ 0', const),
                     FermionOperator('3^ 1', const)]

    assert T1_terms == True_T1_terms



def test_Get_T2_terms_list():
    from openfermion.ops import FermionOperator

    const = 0.25

    ijab_terms = np.array([[3, 2, 1, 0, const],
                         [6, 4, 3, 1, const]])



    T2_terms = Get_T2_terms_list(ijab_terms)

    True_T2_terms = [FermionOperator('3^ 2^ 1 0', const),
                     FermionOperator('6^ 4^ 3 1', const)]

    assert T2_terms == True_T2_terms


def test_dagger_T_list_T1_terms():
    from openfermion.ops import FermionOperator

    const = 0.25

    T1_terms = [FermionOperator('2^ 0', const),
                FermionOperator('3^ 1', const)]

    T1_hermitian_conjugate_True = [FermionOperator('0^ 2', const),
                                  FermionOperator('1^ 3', const)]

    T1_dagger = dagger_T_list(T1_terms)

    assert T1_hermitian_conjugate_True == T1_dagger

def test_dagger_T_list_T2_terms():
    from openfermion.ops import FermionOperator

    const = 0.25

    T2_terms = [FermionOperator('3^ 2^ 1 0', const),
               FermionOperator('6^ 4^ 3 1', const)]

    T2_hermitian_conjugate_True = [FermionOperator('0^ 1^ 2 3', const),
                                   FermionOperator('1^ 3^ 4 6', const)]

    T2_dagger = dagger_T_list(T2_terms)

    assert T2_dagger == T2_hermitian_conjugate_True


def test_dagger_T_list_commutation():
    from openfermion.utils import commutator

    T2_terms = [FermionOperator('3^ 2^ 1 0', const),
                FermionOperator('6^ 4^ 3 1', const)]

    T2_dagger = dagger_T_list(T2_terms)



def test_JW_transform_T1():
    from openfermion.ops import FermionOperator

    const = 0.25

    T1_terms = [FermionOperator('2^ 0', const),
                FermionOperator('3^ 1', const)]

    T1_hermitian_conjugate = [FermionOperator('0^ 2', const),
                              FermionOperator('1^ 3', const)]

    T1_Term_paulis = JW_transform(T1_terms, T1_hermitian_conjugate)

    from openfermion.ops._qubit_operator import QubitOperator

    constant = 0.125j


    # arXiv 1808.10402v2 page 34
    T1_paulis_True = [
        (
            - QubitOperator('X0 Z1 Y2', constant) +
              QubitOperator('Y0 Z1 X2', constant)
    ),

        (
            - QubitOperator('X1 Z2 Y3', constant) +
              QubitOperator('Y1 Z2 X3', constant)
    )

    ]

    assert T1_paulis_True == T1_Term_paulis


def test_JW_transform_T2():
    from openfermion.ops import FermionOperator

    const = 0.25

    T2_terms = [FermionOperator('3^ 2^ 1 0', const)]

    T2_hermitian_conjugate = [FermionOperator('0^ 1^ 2 3', const)]

    T2_Term_paulis = JW_transform(T2_terms, T2_hermitian_conjugate)

    from openfermion.ops._qubit_operator import QubitOperator

    constant = 0.03125j


    # arXiv 1808.10402v2 page 34
    T2_paulis_True = [(
            QubitOperator('X0 X1 X2 Y3', constant) +
            QubitOperator('X0 X1 Y2 X3', constant) -
            QubitOperator('X0 Y1 X2 X3', constant) +
            QubitOperator('X0 Y1 Y2 Y3', constant) -
            QubitOperator('Y0 X1 X2 X3', constant) +
            QubitOperator('Y0 X1 Y2 Y3', constant) -
            QubitOperator('Y0 Y1 X2 Y3', constant) -
            QubitOperator('X3 Y2 Y1 Y0', constant)
    )]

    assert T2_paulis_True == T2_Term_paulis



def test_commutation():

    """
    note:

    commutator:      [a,b] = ab - ba

    anti-commutator: {a,b} = ab + ba

    TODO
    need to think how to use!

    """

    from openfermion.utils import anticommutator
    from openfermion.utils import commutator

    from openfermion.utils import hermitian_conjugated

    x = QubitOperator('X0 X1 X2 Y3', constant)
    y = hermitian_conjugated(x)

    ww = anticommutator(x,y)
    zz = commutator(x,y)

    print('anti-Commutation: ', ww)
    print('Commutation: ', zz)