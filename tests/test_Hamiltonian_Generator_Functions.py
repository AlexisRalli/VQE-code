from quchem.Hamiltonian_Generator_Functions import Hamiltonian
from scipy.sparse import bsr_matrix
import numpy as np
from functools import reduce
from scipy.sparse import kron

import pytest



def test_geometry():
    """
    Standard use test
    """

    X = Hamiltonian('H2')

    pass

def test_Get_Basis_state_in_occ_num_basis():
    # |f_(N −1) , ..., f_1 , f_0> →→→ |q_(N −1) , ..., q_1 , q_0>

    # from openfermion.utils import jw_configuration_state
    # occupied_orbitals = [2, 3]
    # n_qubits = 4
    # jw_configuration_state(occupied_orbitals, n_qubits)

    occupied_orbitals_index_list = [0, 1]
    molecule = 'H2'
    HF_state = [0, 0, 1, 1]

    zero = bsr_matrix(np.array([[1],
                             [0]]))

    one = bsr_matrix(np.array([[0],
                             [1]]))

    OperatorsKeys = {
        '0': zero,
        '1': one,
    }


    HF_state = [OperatorsKeys[str(i)] for i in HF_state[::-1]] # note reverse order!

    tensored_ket = reduce(kron, HF_state)
    tensored_bra = np.transpose(tensored_ket)

    Hamilt = Hamiltonian(molecule,
                         run_scf=1, run_mp2=1, run_cisd=1, run_ccsd=1, run_fci=1,
                         basis='sto-3g',
                         multiplicity=1,
                         geometry=None)
    Hamilt.Get_Molecular_Hamiltonian()
    ket, bra = Hamilt.Get_Basis_state_in_occ_num_basis(occupied_orbitals_index_list)

    assert np.array_equal(ket.todense(), tensored_ket.todense()) and np.array_equal(bra.todense(), tensored_bra.todense())


