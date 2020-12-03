from quchem.Hamiltonian_Generator_Functions import Hamiltonian
import numpy as np
# in terminal type: py.test -v
import pytest



@pytest.fixture
def supply_H2_Hamiltonian():
    """
    Give Hamiltonian fixture... Aka don't have to run Hamiltonian class in each method...
    especially expensive psi4 calculation!
    """
    molecule = 'H2'
    Hamilt = Hamiltonian(molecule,
                         run_scf=1, run_mp2=1, run_cisd=1, run_ccsd=1, run_fci=1,
                         basis='sto-3g',
                         multiplicity=1,
                         geometry=None)
    Hamilt.Run_Psi4()
    return Hamilt

def test_geometry_standard(supply_H2_Hamiltonian):
    """
    Standard use test
    """
    Hamilt = supply_H2_Hamiltonian
    Hamilt.Get_Geometry()

    true_geometry = [('H', (2, 0, 0)), ('H', (3, 0, 0))]

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

def test_Run_Psi4(supply_H2_Hamiltonian):
    """
    Standard use test
    """
    Hamilt = supply_H2_Hamiltonian
    # Hamilt.Run_Psi4()
    # already run in fixure!
    assert Hamilt.molecule is not None

def test_Get_ia_and_ijab_terms_CC_ops(supply_H2_Hamiltonian):
    Hamilt = supply_H2_Hamiltonian
    SQ_CC_ops, THETA_params = Hamilt.Get_ia_and_ijab_terms(Coupled_cluser_param=True)

    from openfermion.ops import FermionOperator
    expected = [FermionOperator('0^ 2', -1) + FermionOperator('2^ 0', 1),
                FermionOperator('1^ 3', -1) + FermionOperator('3^ 1', 1),
                FermionOperator('0^ 1^ 2 3', -1) + FermionOperator('3^ 2^ 1 0', 1),
                ]
    print(SQ_CC_ops)

    assert SQ_CC_ops == expected

def test_Get_Basis_state_in_occ_num_basis_using_HF_JW(supply_H2_Hamiltonian):
    """
    Test funct ability using JW ground state
    """
    Hamilt = supply_H2_Hamiltonian

    occ_orb_index = [i for i in range(Hamilt.molecule.n_electrons)]
    reference_ket, reference_bra = Hamilt.Get_Basis_state_in_occ_num_basis(occupied_orbitals_index_list=occ_orb_index)

    from openfermion.utils._sparse_tools import jw_hartree_fock_state
    HF_state_vec = jw_hartree_fock_state(Hamilt.molecule.n_electrons, Hamilt.molecule.n_qubits)
    HF_state_ket = HF_state_vec.reshape([len(HF_state_vec), 1])

    assert np.array_equal(reference_ket.toarray(), HF_state_ket)

def test_Convert_basis_state_to_occ_num_basis(supply_H2_Hamiltonian):
    """
    blah
    """
    Hamilt = supply_H2_Hamiltonian

    from openfermion.utils._sparse_tools import jw_hartree_fock_state
    HF_state_vec = jw_hartree_fock_state(Hamilt.molecule.n_electrons, Hamilt.molecule.n_qubits)
    HF_state_ket = HF_state_vec.reshape([len(HF_state_vec), 1])

    output = Hamilt.Convert_basis_state_to_occ_num_basis(HF_state_ket)

    expected = ['1' if i in np.arange(0,Hamilt.molecule.n_electrons,1) else '0' for i in range(Hamilt.molecule.n_qubits)]

    assert output == expected

def test_Get_FCI_from_MolecularHamialtonian(supply_H2_Hamiltonian):
    """
    blah
    """
    Hamilt = supply_H2_Hamiltonian
    E_FCI = Hamilt.Get_FCI_from_MolecularHamialtonian()

    assert np.isclose(E_FCI.real, Hamilt.molecule.fci_energy, rtol=1e-09, atol=0.0)


### test:
# from numpy import kron
# from functools import reduce
# zero = np.array([[1], [0]])
# one = np.array([[0], [1]])
# STATE = [zero, one, zero, zero]
# STATE_vec = reduce(kron, STATE)
# print(Convert_basis_state_to_occ_num_basis(STATE_vec))





# def test_Get_Basis_state_in_occ_num_basis(supply_H2_Hamiltonian):
#     # |f_(N −1) , ..., f_1 , f_0> →→→ |q_(N −1) , ..., q_1 , q_0>
#
#     # from openfermion.utils import jw_configuration_state
#     # occupied_orbitals = [2, 3]
#     # n_qubits = 4
#     # jw_configuration_state(occupied_orbitals, n_qubits)
#
#     Hamilt = supply_H2_Hamiltonian
#     Hamilt.Get_Molecular_Hamiltonian()
#     ket, bra = Hamilt.Get_Basis_state_in_occ_num_basis(occupied_orbitals_index_list)
#
#
#     occupied_orbitals_index_list = [i for i in range(Hamilt.molecule.n_electrons)]
#     HF_state = [1 for _ in range(Hamilt.molecule.n_electrons)] + \
#                [0 for _ in range(Hamilt.molecule.n_qubits-Hamilt.molecule.n_electrons)] # note ORDER (occupied first)
#
#     zero = bsr_matrix(np.array([[1],
#                                 [0]]))
#     one = bsr_matrix(np.array([[0],
#                                [1]]))
#
#     OperatorsKeys = {
#         '0': zero,
#         '1': one,
#     }
#     HF_state = [OperatorsKeys[str(i)] for i in HF_state] # note reverse order!
#
#
#     tensored_ket = reduce(kron, HF_state)
#     tensored_bra = np.transpose(tensored_ket)
#
#
#
#     assert np.array_equal(ket.todense(), tensored_ket.todense()) and \
#            np.array_equal(bra.todense(), tensored_bra.todense())
#
#
# #TODO
# # def test_Get_CCSD_Amplitudes()
#
# def test_Get_Molecular_Hamiltonian():
#     Hamilt.Get_Molecular_Hamiltonian()
#
#     mol_ham_mat = Hamilt.MolecularHamiltonianMatrix
