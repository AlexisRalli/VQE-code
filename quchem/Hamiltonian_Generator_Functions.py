import numpy as np
import scipy

from openfermion.hamiltonians import MolecularData
from openfermionpsi4 import run_psi4


class Hamiltonian():
    """

    The UCC_Terms object calculates and retains all the unitary coupled cluster terms.

    Args:
        MoleculeName (str): Name of Molecule
        run_scf (int): Boolean to run SCF calculation.
        run_mp2 (int): Boolean to run MP2 calculation.
        run_cisd (int): Boolean to run CISD calculation.
        run_ccsd (int): Boolean to run CCSD calculation.
        run_fci (int): Boolean to FCI calculation.
        multiplicity_(int): Multiplicity of molecule (1=singlet, 2=doublet ... etc)
        geometry (list, optional): Geometry of Molecule (if None will find it online)
        delete_input (bool, optional): Optional boolean to delete psi4 input file.
        delete_output: Optional boolean to delete psi4 output file

    Attributes:
        molecule (openfermion.hamiltonians._molecular_data.MolecularData): An instance of the MolecularData class

        molecule.single_cc_amplitudes (numpy.array): h_pq (n_qubits x n_qubits) numpy array
        molecule.double_cc_amplitudes (numpy.array): h_pqrs (n_qubits x n_qubits x n_qubits x n_qubits) numpy array

        singles_hamiltonian (numpy.ndarray): h_pq (n_qubits x n_qubits) matrix.
        doubles_hamiltonian (numpy.ndarray): h_pqrs (n_qubits x n_qubits x n_qubits x n_qubits) matrix
        MolecularHamiltonian (openfermion.ops._interaction_operator.InteractionOperator):
        MolecularHamiltonianMatrix (scipy.sparse.csc.csc_matrix): Sparse matrix of Molecular Hamiltonian
        num_theta_parameters(int): Number of theta parameters in for ia_and_ijab_terms

    """
    def __init__(self, MoleculeName,
                 run_scf = 1, run_mp2 = 1, run_cisd = 1, run_ccsd = 1, run_fci=1,
                 basis='sto-3g', multiplicity=1, geometry=None, delete_input=False, delete_output=False):

        self.MoleculeName = MoleculeName
        self.run_scf = bool(run_scf)
        self.run_mp2 = bool(run_mp2)
        self.run_cisd = bool(run_cisd)
        self.run_ccsd = bool(run_ccsd)
        self.run_fci = bool(run_fci)
        self.geometry = geometry
        self.multiplicity = multiplicity
        self.basis = basis
        self.molecule = None
        self.delete_input = delete_input
        self.delete_output = delete_output
        self.num_theta_parameters = None
        self.MolecularHamiltonianMatrix = None

    def Run_Psi4(self):

        if self.geometry is None:
            self.Get_Geometry()

        # input
        self.molecule = MolecularData(
            self.geometry,
            self.basis,
            self.multiplicity,
            description=self.MoleculeName)

        #output file
        self.molecule.filename = self.MoleculeName

        # Run Psi4.
        self.molecule = run_psi4(self.molecule,
                            run_scf=self.run_scf,
                            run_mp2=self.run_mp2,
                            run_cisd=self.run_cisd,
                            run_ccsd=self.run_ccsd,
                            run_fci=self.run_fci,
                            delete_input=False,
                            delete_output=False)
        # note template: ~/envs/ENV_NAME/lib/python3.7/site-packages/openfermionpsi4 : _psi4_template
        # https://github.com/quantumlib/OpenFermion-Psi4/blob/master/openfermionpsi4/_psi4_template

    def Get_Geometry(self):

        from openfermion.utils import geometry_from_pubchem
        self.geometry = geometry_from_pubchem(self.MoleculeName)

        if self.geometry is None:
            raise ValueError('Unable to find molecule in the PubChem database.')

    def PrintInfo(self):
        if self.molecule is None:
            self.Run_Psi4()

        print('Geometry: ', self.geometry)
        print('No Qubits: ', self.molecule.n_qubits)
        print('No. Spin Orbitals: ', self.molecule.n_orbitals * 2)
        print('multiplicity: ', self.multiplicity)

        print('HF Energy: ', self.molecule.hf_energy)
        print('CCSD: ', self.molecule.ccsd_energy)
        print('FCI: ', self.molecule.fci_energy)

    def Get_Molecular_Hamiltonian(self, Get_H_matrix=False):
        """
        Get sparse matrix of molecular Hamiltonian

        Attributes:
            singles_hamiltonian (numpy.ndarray): h_pq (n_qubits x n_qubits) matrix.
            doubles_hamiltonian (numpy.ndarray): h_pqrs (n_qubits x n_qubits x n_qubits x n_qubits) matrix
            MolecularHamiltonian (openfermion.ops._interaction_operator.InteractionOperator):
            MolecularHamiltonianMatrix (scipy.sparse.csc.csc_matrix): Sparse matrix of Molecular Hamiltonian


        """
        if self.molecule is None:
            self.Run_Psi4()

        # H = constant + ∑_pq (h_pq a†_p a_q) + ∑_pqrs (h_pqrs a†_p a†_q a_r a_s)
        self.MolecularHamiltonian = self.molecule.get_molecular_hamiltonian() # instance of the MolecularOperator class
        self.singles_hamiltonian = self.MolecularHamiltonian.one_body_tensor # h_pq (n_qubits x n_qubits numpy array)
        self.doubles_hamiltonian = self.MolecularHamiltonian.two_body_tensor # h_pqrs (n_qubits x n_qubits x n_qubits x n_qubits numpy array


        # Get Matrix Form of QubitHamiltonian
        if Get_H_matrix is True:
            from openfermion.transforms import get_sparse_operator
            self.MolecularHamiltonianMatrix = get_sparse_operator(self.MolecularHamiltonian)

    def Get_CCSD_Amplitudes(self):
        """
        Parse coupled cluster singles and doubles amplitudes from psi4 file
        where: H = constant + ∑_pq (h_pq a†_p a_q) + ∑_pqrs (h_pqrs a†_p a†_q a_r a_s)
        note that:
            - single amplitudes: h_pq is a (n_qubits x n_qubits) numpy array
            - doubles amplitudes: h_pqrs is a (n_qubits x n_qubits x n_qubits x n_qubits) numpy array
        """

        from openfermionpsi4._psi4_conversion_functions import parse_psi4_ccsd_amplitudes
        # https://github.com/quantumlib/OpenFermion-Psi4/blob/master/openfermionpsi4/_psi4_conversion_functions.py
        if self.molecule is None:
            self.Run_Psi4()

        self.molecule.single_cc_amplitudes, self.molecule.double_cc_amplitudes = (
                                                                    parse_psi4_ccsd_amplitudes(
                                                                        2 * self.molecule.n_orbitals,
                                                                        self.molecule.get_n_alpha_electrons(),
                                                                        self.molecule.get_n_beta_electrons(),
                                                                        self.molecule.filename + ".out"))

    def Get_FCI_from_MolecularHamialtonian(self):

        if self.MolecularHamiltonianMatrix is None:
            self.Get_Molecular_Hamiltonian(Get_H_matrix=True)

        from scipy.sparse.linalg import eigs
        eig_values, eig_vectors = eigs(self.MolecularHamiltonianMatrix)
        FCI_Energy = min(eig_values)

        if not np.isclose(FCI_Energy.real, self.molecule.fci_energy, rtol=1e-09, atol=0.0):
            raise ValueError('Calculated FCI energy from Moleular Hamiltonian Operator not equivalent to PSI4 calculation')
        return FCI_Energy

    def Get_Basis_state_in_occ_num_basis(self, occupied_orbitals_index_list=None):
        """

        Method to obtain basis state under JW transform of state defined in occupation number basis.
        e.g. for H2 under the Jordan Wigner transfrom has |HF> = |0011> in occ no. basis
        occupied_orbitals_index_list = [0,1] <- as first orbitals occupied

        These outputs (|HF> and <HF|) can be used with MolecularHamiltonianMatrix!.

        Args:
            occupied_orbitals_index_list (list, optional): list of orbital indices that are OCCUPIED

        returns:
            reference_ket (scipy.sparse.csr.csr_matrix): Sparse matrix of KET corresponding to occ no basis state under
                                                         JW transform
            reference_bra (scipy.sparse.csr.csr_matrix): Sparse matrix of BRA corresponding to occ no basis state under
                                                         JW transform

        """

        from openfermion import jw_configuration_state

        if occupied_orbitals_index_list is None:
            reference_ket = scipy.sparse.csc_matrix(jw_configuration_state(list(range(0, self.molecule.n_electrons)),
                                                   self.molecule.n_qubits)).transpose()
        else:
            reference_ket = scipy.sparse.csc_matrix(jw_configuration_state(occupied_orbitals_index_list,
                                                                           self.molecule.n_qubits)).transpose()
        reference_bra = reference_ket.transpose().conj()
        return reference_ket, reference_bra

        # To be used as follows:
        # hamiltonian_ket = self.MolecularHamiltonianMatrix.dot(reference_ket)

    def Convert_basis_state_to_occ_num_basis(self, state):
        """

        For a state defined by a basis state vector... get state in occupation number basis. Note doesn't work for
        entangled states!



        Args:
            state (numpy.ndarray): dense numpy array of basis state

        returns:
            state_list (list): List of qubit values in order

        e.g.
        np.arrayarray([[0],
                   [0],
                   [0],
                   [1],
                   [0],
                   [0],
                   [0],
                   [0]])

        output = ['0', '1', '1']

        """
        Number_Qubits = int(np.log2(int(state.shape[0])))

        state_list = []

        for _ in range(Number_Qubits):
            length = int(state.shape[0])

            position_1 = np.where(state == 1)[0]

            if position_1 < length / 2:
                # single_q = np.array([[1], [0]])
                single_q = '0'
                state = state[0:int(length / 2), :]
            else:
                # single_q = np.array([[0], [1]])
                single_q = '1'
                state = state[int(length / 2)::, :]
            state_list.append(single_q)

        return state_list

    def Get_NOON(self):
        """
        See https://journals.aps.org/prx/pdf/10.1103/PhysRevX.8.031022 appendix C for further details

        Diagonalizing the 1-RDM (in the canonical orbital basis from a CISD calculation) gives the 1-RDM
        fermionic natural molecular orbitals (NMO) basis, which are arranged as "spin-up, spin-down, spin-up, spin-down..."

        The value of each entry gives the natural orbital occupation number (NOON). Therefore orbitals with a
        small NOON can be assumed to be UNFILLED and REMOVED from the Hamiltonian!


        returns:
            NOON_spins_combined (np.array): NOON with same spin-up and spin-down spatial orbitals combined

         e.g. for LiH with a bond length of 1.45 A:
         NMO_basis =
                        array([
                               9.99961301e-01, 9.81031486e-01,
                               9.99961301e-01, 9.81031486e-01,
                               1.72719307e-02, 1.72719307e-02,
                               2.38704909e-05, 2.38704909e-05,
                               8.55705967e-04, 8.55705967e-04,
                               8.55705967e-04, 8.55705967e-04
                             ])

         NOON_spins_combined =
                               array([
                                   1.98099279e+00,
                                   1.98099279e+00,
                                   3.45438614e-02,
                                   4.77409818e-05,
                                   1.71141193e-03,
                                   1.71141193e-03
                               ])
         therefore here can see that for orbitals with index 0 and 1 have NOON close to TWO, therefore can consider it
         always doubly occupied... so can remove any terms in the hamiltonian containing: a†0 ,a0, a†1, a1.

         Also have a look at indices 6,7 ... occupation is: 4.77409818e-05 ... VERY SMALL NOON... therefore can assume
         these orbitals are never occupied. Again can remove the fermion operators from the Hamiltonian too:
         a†6 ,a6, a†8, a7.

        """
        if self.molecule is None:
            self.Run_Psi4()

        one_RDM = self.molecule.cisd_one_rdm

        # THIS is wrong:
        # from numpy import diag
        # NMO_basis = diag(one_RDM)

        from numpy.linalg import eig
        NMO_basis, eig_vectors = eig(one_RDM)

        # adding each pair of entries
        NOON_spins_combined = np.add.reduceat(NMO_basis, np.arange(0, len(NMO_basis), 2))

        return NMO_basis, NOON_spins_combined

    def Get_Fermionic_Hamiltonian(self):

        """
        Returns the second quantised Fermionic Molecular Hamiltonian of the Molecular Hamiltonian

        e.g. H = h00 a†0a0 + h11a†1a1 + h22a†2a2 + h33a†3a3 + h0110 a†0a†1a1a0 +h2332a†2a†3a3a2 + ... etc etc

        note can get integrals (h_ia and h_ijab) from Get_CCSD_Amplitudes method of Hamiltonian class!

        returns:
            FermionicHamiltonian (openfermion.ops._fermion_operator.FermionOperator): Fermionic Operator

        e.g. for H2:
                             0.7151043387432434 [] +
                            -1.2533097864345657 [0^ 0] +
                            0.3373779633738658 [0^ 0^ 0 0] +
                            0.09060523101737843 [0^ 0^ 2 2] +
                            0.3373779633738658 [0^ 1^ 1 0] +
                            0.09060523101737843 [0^ 1^ 3 2] + ... etc ... etc

        """

        from openfermion.transforms import get_fermion_operator
        FermionicHamiltonian = get_fermion_operator(self.MolecularHamiltonian)

        return FermionicHamiltonian

    def Get_Qubit_Hamiltonian(self, threshold=None, transformation='JW'):

        """
        Returns the second quantised Qubit Molecular Hamiltonian of the Molecular Hamiltonian using the
        JORDAN WIGNER transformation. First gets the fermionic Hamiltonian and than performs JW.

        e.g. H = h0 I + h1 Z0 + h2 Z1 +h3 Z2 + h4 Z3 + h5 Z0Z1 ... etc etc
        note can get integrals (h_ia and h_ijab) from Get_CCSD_Amplitudes method of Hamiltonian class!

        args:
            threshold (optional, float): gives threshold of terms to ignore... e.g. the term
                                        (0.00003+0j) [Y0 X1 X2 Y3]] would be ignored if threshold = 1e-2
        returns:
            QubitHamiltonian (openfermion.ops._qubit_operator.QubitOperator): Qubit Operator

        e.g. for H2:
                             (-0.09706626861762624+0j) [] +
                            (-0.04530261550868928+0j) [X0 X1 Y2 Y3] +
                            (0.04530261550868928+0j) [X0 Y1 Y2 X3] +
                            (0.04530261550868928+0j) [Y0 X1 X2 Y3] +
                            (-0.04530261550868928+0j) [Y0 Y1 X2 X3] +
                            (0.17141282639402405+0j) [Z0] + ... etc etc

        """


        FermionicHamiltonian = self.Get_Fermionic_Hamiltonian()

        if transformation == 'JW':
            from openfermion.transforms import jordan_wigner
            QubitHamiltonian = jordan_wigner(FermionicHamiltonian)
        elif transformation == 'BK':
            from openfermion.transforms import bravyi_kitaev
            QubitHamiltonian = bravyi_kitaev(FermionicHamiltonian)


        if threshold is None:
            return QubitHamiltonian
        else:
            from openfermion.ops import QubitOperator
            reduced_QubitHamiltonian = QubitOperator()
            for key in QubitHamiltonian.terms:
                if np.abs(QubitHamiltonian.terms[key]) > threshold:
                    reduced_QubitHamiltonian += QubitOperator(key, QubitHamiltonian.terms[key])
            return reduced_QubitHamiltonian

    def Get_sparse_Qubit_Hamiltonian_matrix(self, QubitOperator):
        """
        Get sparse matrix of qubit Hamiltonian

        Args:
            QubitOperator (openfermion.ops._qubit_operator.QubitOperator): Qubit operator

        Returns:
            16x16 sparse matrix

        e.g.
        input:
            (-0.09706626861762581+0j) [] +
            (-0.045302615508689394+0j) [X0 X1 Y2 Y3] +
            (0.045302615508689394+0j) [X0 Y1 Y2 X3] +
            (0.045302615508689394+0j) [Y0 X1 X2 Y3] +
            (-0.045302615508689394+0j) [Y0 Y1 X2 X3] +
            (0.17141282639402383+0j) [Z0]
        """
        from openfermion import qubit_operator_sparse
        return qubit_operator_sparse(QubitOperator)
