from Ansatz_Generator_Functions import *
import pytest
# in terminal type: py.test -v

def test_Get_Occupied_and_Unoccupied_sites_H2():
    HF_State = [0,0,1,1]

    up_occ_true = [0]  # spin up
    down_occ_true = [1]  # spin down

    up_unocc_true = [2]
    down_unocc_true = [3]

    up_occ, down_occ, up_unocc, down_unocc = Get_Occupied_and_Unoccupied_sites(HF_State)

    assert (up_occ, down_occ, up_unocc, down_unocc) == (up_occ_true, down_occ_true, up_unocc_true, down_unocc_true)

def test_Get_Occupied_and_Unoccupied_sites_H2O():

    HF_State = [0,0,0,0,1,1,1,1,1,1,1,1,1,1]

    up_occ_true = [0,2,4,6,8] # spin up
    down_occ_true = [1,3,5,7,9] # spin down

    up_unocc_true = [10, 12]
    down_unocc_true = [11, 13]

    up_occ, down_occ, up_unocc, down_unocc = Get_Occupied_and_Unoccupied_sites(HF_State)

    assert (up_occ, down_occ, up_unocc, down_unocc) == (up_occ_true, down_occ_true, up_unocc_true, down_unocc_true)

def test_Get_Occupied_and_Unoccupied_sites_DIFFERENT_ORDERING():

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

    # H2 HF_state = [0,0,1,1]

    up_occ = [0]
    down_occ = [1]
    up_unocc = [2]
    down_unocc = [3]


    ia_terms_true = np.array([[2,0,0.25],
                              [3, 1, 0.25]])

    ijab_terms_true = np.array([2, 3, 0,1, 0.25])

    ia_terms, ijab_terms = Get_ia_and_ijab_terms(up_occ, down_occ, up_unocc, down_unocc)

    assert (ia_terms_true.all(), ijab_terms_true.all()) == (ia_terms.all(), ijab_terms.all()) # NEED TO CHECK THIS AND MAKE SURE IT IS WORKING PROPERLY!

def test_Get_ia_and_ijab_terms_4_sites():

    # HF_State = [0,0,0,0,1,1,1,1]

    up_occ = [0,2]
    down_occ = [1,3]
    up_unocc = [4,6]
    down_unocc = [5,7]


    ia_terms_true = np.array([
                              [4,0,0.25],
                              [6, 0, 0.25],

                              [4, 2, 0.25],
                              [6, 2, 0.25],

                              [5, 1, 0.25],
                              [7, 1, 0.25],

                              [5, 3, 0.25],
                              [7, 4, 0.25]
                              ])

    ijab_terms_true = np.array([                         # need to finish WRITING THIS ARRAY OUT!!!!
                                [4, 6, 0, 2, 0.25],
                                #[6, 4, 0, 2, 0.25],

                                #[5, 7, 1, 3, 0.25],
                                [7, 5, 1, 3, 0.25],

                                [6, 5, 2, 1, 0.25],
                                [6, 7, 0, 1, 0.25],
                                [4, 5, 2, 3, 0.25],
                                [6, 7, 2, 3, 0.25],

                                [4, 5, 0, 1, 0.25],
                                [6, 7, 0, 1, 0.25],
                                [4, 5, 2, 3, 0.25],
                                [6, 7, 2, 3, 0.25],

    ])



    ia_terms, ijab_terms = Get_ia_and_ijab_terms(up_occ, down_occ, up_unocc, down_unocc)

    assert (ia_terms_true.all(), ijab_terms_true.all()) == (ia_terms.all(), ijab_terms.all()) # NEED TO CHECK THIS AND MAKE SURE IT IS WORKING PROPERLY!


# HF_State = [0,0,0,0,1,1,1,1]
# up_occ, down_occ, up_unocc, down_unocc = Get_Occupied_and_Unoccupied_sites(HF_State)
# ia, ijab = Get_ia_and_ijab_terms(up_occ, down_occ, up_unocc, down_unocc)