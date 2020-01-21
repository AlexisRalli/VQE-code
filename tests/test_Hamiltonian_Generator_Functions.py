from quchem.Hamiltonian_Generator_Functions import Hamiltonian
from scipy.sparse import bsr_matrix
import numpy as np
from functools import reduce
from scipy.sparse import kron

import pytest

molecule = 'H2'
Hamilt = Hamiltonian(molecule,
                     run_scf=1, run_mp2=1, run_cisd=1, run_ccsd=1, run_fci=1,
                     basis='sto-3g',
                     multiplicity=1,
                     geometry=None)

true_geometry = [('H', (2, 0, 0)), ('H', (3, 0, 0))]

def test_geometry_standard():
    """
    Standard use test
    """

    Hamilt.Get_Geometry()

    assert Hamilt.geometry == true_geometry

def test_Get_Geometry_WRONG_NAME():
    mol_name = 'NOT a molecule'
    H_test = Hamiltonian(mol_name,
                         run_scf=1, run_mp2=1, run_cisd=1, run_ccsd=1, run_fci=1,
                         basis='sto-3g',
                         multiplicity=1,
                         geometry=None)

    with pytest.raises(ValueError) as excinfo:
        assert H_test.Get_Geometry() in excinfo.value

def test_Run_Psi4():
    """
    Standard use test
    """
    Hamilt.Run_Psi4()
    assert Hamilt.molecule is not None

def test_Get_Basis_state_in_occ_num_basis():
    # |f_(N −1) , ..., f_1 , f_0> →→→ |q_(N −1) , ..., q_1 , q_0>

    # from openfermion.utils import jw_configuration_state
    # occupied_orbitals = [2, 3]
    # n_qubits = 4
    # jw_configuration_state(occupied_orbitals, n_qubits)

    occupied_orbitals_index_list = [i for i in range(Hamilt.molecule.n_electrons)]
    HF_state = [1 for _ in range(Hamilt.molecule.n_electrons)] + \
               [0 for _ in range(Hamilt.molecule.n_qubits-Hamilt.molecule.n_electrons)] # note ORDER (occupied first)

    zero = bsr_matrix(np.array([[1],
                                [0]]))
    one = bsr_matrix(np.array([[0],
                               [1]]))

    OperatorsKeys = {
        '0': zero,
        '1': one,
    }
    HF_state = [OperatorsKeys[str(i)] for i in HF_state] # note reverse order!


    tensored_ket = reduce(kron, HF_state)
    tensored_bra = np.transpose(tensored_ket)

    Hamilt.Get_Molecular_Hamiltonian()
    ket, bra = Hamilt.Get_Basis_state_in_occ_num_basis(occupied_orbitals_index_list)

    assert np.array_equal(ket.todense(), tensored_ket.todense()) and \
           np.array_equal(bra.todense(), tensored_bra.todense())


#TODO
# def test_Get_CCSD_Amplitudes()

def test_Get_Molecular_Hamiltonian():
    Hamilt.Get_Molecular_Hamiltonian()

    mol_ham_mat = Hamilt.MolecularHamiltonianMatrix
