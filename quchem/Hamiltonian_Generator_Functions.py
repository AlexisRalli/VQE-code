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

    def Get_ia_and_ijab_terms(self, Coupled_cluser_param=False, filter_small_terms = False): #TODO could add MP2 param option to initialise theta with MP2 amplitudes (rather than coupled cluster only option)
        """

        Get ia and ijab terms as fermionic creation and annihilation operators which perform UCCSD.
        These ia and ijab terms are standard single and double excitation terms (aka only occupied -> unoccupied transitions allowed)
        (faster and only marginally less accurate.)
        #TODO can add method to get pqrs terms
        #TODO these are all non-degenerate excitations which can possibly non-zero, including nocc->nocc, occ->occ, and spin-flips.
        #TODO EXPENSIVE, but will likely  get a slightly better answer.

        Args:
            Coupled_cluser_param (bool, optional): Whether to use CC calculated amplitdues to initialise theta_angles
            filter_small_terms (bool, optional):  Whether to filter small terms in Hamiltonian (threshold currently hardcoded)

        returns:
            Sec_Quant_CC_ops (list): list of FermionOperators (openfermion.ops._fermion_operator.FermionOperator)
            theta_parameters (list): list of theta values (parameterisation of excitation amplitudes)

         e.g. for H2:
         Sec_Quant_CC_ops=  [
                             -1.0[0 ^ 2] + 1.0[2 ^ 0],            # -(a†0 a2) + (a†2 a0)
                             -1.0[1 ^ 3] + 1.0[3 ^ 1],            # -(a†1 a3) + (a†3 a1)
                             -1.0[0 ^ 1 ^ 2 3] + 1.0 [3^ 2^ 1 0]  # -(a†0 a†1 a2 a3) + a†3 a†2 a1 a0)
                            ]
        theta_parameters = [0,0,0]

        """


        if self.molecule is None:
            self.Run_Psi4()

        if Coupled_cluser_param is True:
            self.Get_CCSD_Amplitudes()

        from openfermion.ops import FermionOperator
        self.No_spin_oribtals = int(self.molecule.n_orbitals*2)


        orbitals_index = range(0, self.No_spin_oribtals)
        alph_occs = [k for k in orbitals_index if k % 2 == 0 and k < self.molecule.n_electrons] # spin up occupied
        beta_occs = [k for k in orbitals_index if k % 2 == 1 and k < self.molecule.n_electrons] # spin down UN-occupied
        alph_noccs = [k for k in orbitals_index if k % 2 == 0 and k >=self.molecule.n_electrons] # spin down occupied
        beta_noccs = [k for k in orbitals_index if k % 2 == 1 and k >= self.molecule.n_electrons] # spin up UN-occupied

        Sec_Quant_CC_ops = [] # second quantised CC operators
        theta_parameters =[]

        # SINGLE electron excitation: spin UP transition
        for i in alph_occs:
            for a in alph_noccs:
                if filter_small_terms is True:
                # uses Hamiltonian to ignore small terms!
                    if abs(self.singles_hamiltonian[i][a]) > 1e-8 or abs(self.singles_hamiltonian[a][i]) > 1e-8:
                        one_elec = FermionOperator(((a, 1), (i, 0))) - FermionOperator(((i, 1), (a, 0)))
                        if Coupled_cluser_param is True:
                            theta_parameters.append(self.molecule.single_cc_amplitudes[a][i])
                        else:
                            theta_parameters.append(0)

                        Sec_Quant_CC_ops.append(one_elec)
                else:
                    # NO filtering
                    one_elec = FermionOperator(((a, 1), (i, 0))) - FermionOperator(((i, 1), (a, 0)))
                    if Coupled_cluser_param is True:
                        theta_parameters.append(self.molecule.single_cc_amplitudes[a][i])
                    else:
                        theta_parameters.append(0)

                    Sec_Quant_CC_ops.append(one_elec)

        # SINGLE electron excitation: spin DOWN transition
        for i in beta_occs:
            for a in beta_noccs:
                if filter_small_terms is True:
                    # uses Hamiltonian to ignore small terms!
                    if abs(self.singles_hamiltonian[i][a]) > 1e-8 or abs(self.singles_hamiltonian[a][i]) > 1e-8:
                        one_elec = FermionOperator(((a, 1), (i, 0))) - FermionOperator(((i, 1), (a, 0)))
                        if Coupled_cluser_param is True:
                            theta_parameters.append(self.molecule.single_cc_amplitudes[a][i])
                        else:
                            theta_parameters.append(0)

                        Sec_Quant_CC_ops.append(one_elec)
                else:
                    # NO filtering
                    one_elec = FermionOperator(((a, 1), (i, 0))) - FermionOperator(((i, 1), (a, 0)))
                    if Coupled_cluser_param is True:
                        theta_parameters.append(self.molecule.single_cc_amplitudes[a][i])
                    else:
                        theta_parameters.append(0)

                    Sec_Quant_CC_ops.append(one_elec)

        # DOUBLE excitation: UP + UP
        for i in alph_occs:
            for j in [k for k in alph_occs if k > i]:
                for a in alph_noccs:
                    for b in [k for k in alph_noccs if k > a]:

                        if filter_small_terms is True:
                            # uses Hamiltonian to ignore small terms!
                            if abs(self.doubles_hamiltonian[j][i][a][b]) > 1e-8 or abs(self.doubles_hamiltonian[b][a][i][j]) > 1e-8:
                                two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
                                           FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))
                                if Coupled_cluser_param is True:
                                    theta_parameters.append(self.molecule.double_cc_amplitudes[b][a][j][i])
                                else:
                                    theta_parameters.append(0)
                            Sec_Quant_CC_ops.append(two_elec)
                        else:
                            # NO filtering
                            two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
                                       FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))

                            if Coupled_cluser_param is True:
                                theta_parameters.append(self.molecule.double_cc_amplitudes[b][a][j][i])
                            else:
                                theta_parameters.append(0)

                            Sec_Quant_CC_ops.append(two_elec)

        # DOUBLE excitation: DOWN + DOWN
        for i in beta_occs:
            for j in [k for k in beta_occs if k > i]:
                for a in beta_noccs:
                    for b in [k for k in beta_noccs if k > a]:

                        if filter_small_terms is True:
                            # uses Hamiltonian to ignore small terms!
                            if abs(self.doubles_hamiltonian[j][i][a][b]) > 1e-8 or abs(self.doubles_hamiltonian[b][a][i][j]) > 1e-8:
                                two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
                                           FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))
                                if Coupled_cluser_param is True:
                                    theta_parameters.append(self.molecule.double_cc_amplitudes[b][a][j][i])
                                else:
                                    theta_parameters.append(0)
                            Sec_Quant_CC_ops.append(two_elec)
                        else:
                            # NO filtering
                            two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
                                       FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))

                            if Coupled_cluser_param is True:
                                theta_parameters.append(self.molecule.double_cc_amplitudes[b][a][j][i])
                            else:
                                theta_parameters.append(0)

                            Sec_Quant_CC_ops.append(two_elec)

        # DOUBLE excitation: up + DOWN
        for i in alph_occs:
            for j in [k for k in beta_occs if k > i]:
                for a in alph_noccs:
                    for b in [k for k in beta_noccs if k > a]:

                        if filter_small_terms is True:
                            # uses Hamiltonian to ignore small terms!
                            if abs(self.doubles_hamiltonian[j][i][a][b]) > 1e-8 or abs(self.doubles_hamiltonian[b][a][i][j]) > 1e-8:
                                two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
                                           FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))
                                if Coupled_cluser_param is True:
                                    theta_parameters.append(self.molecule.double_cc_amplitudes[b][a][j][i])
                                else:
                                    theta_parameters.append(0)
                            Sec_Quant_CC_ops.append(two_elec)
                        else:
                            # NO filtering
                            two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
                                       FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))

                            if Coupled_cluser_param is True:
                                theta_parameters.append(self.molecule.double_cc_amplitudes[b][a][j][i])
                            else:
                                theta_parameters.append(0)

                            Sec_Quant_CC_ops.append(two_elec)

        self.num_theta_parameters = len(theta_parameters)
        return Sec_Quant_CC_ops, theta_parameters

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
        elif:
            from openfermion.transforms import bravyi_kitaev
            QubitHamiltonian = bravyi_kitaev(FermionicHamiltonian)

        if transformation == 'BK':
            from openfermion.ops import QubitOperator
            reduced_QubitHamiltonian = QubitOperator()
            for key in QubitHamiltonian.terms:
                if np.abs(QubitHamiltonian.terms[key]) > threshold:
                    reduced_QubitHamiltonian += QubitOperator(key, QubitHamiltonian.terms[key])
            return reduced_QubitHamiltonian
        else:
            return QubitHamiltonian

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




class Hamiltonian_Transforms():
    """

    The Hamiltonian_Transforms object manipulates outputs from Hamiltonian class
    Can get:
        - Fermionic Molecular Hamiltonian
        - Qubit Molecular Hamiltonian under JW transform
        - Jordan Wigner CC matrices

    Args:
        MolecularHamiltonian (openfermion.ops._interaction_operator.InteractionOperator):
        Sec_Quant_CC_ops (list): list of FermionOperators (openfermion.ops._fermion_operator.FermionOperator)
        n_qubits (int): Number of qubits

    """
    def __init__(self, MolecularHamiltonian, Sec_Quant_CC_ops, n_qubits):
        self.MolecularHamiltonian = MolecularHamiltonian
        self.Sec_Quant_CC_ops = Sec_Quant_CC_ops
        self.n_qubits = n_qubits

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

    def Get_Qubit_Hamiltonian_JW(self, threshold=None):

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

        from openfermion.transforms import jordan_wigner
        FermionicHamiltonian = self.Get_Fermionic_Hamiltonian()
        QubitHamiltonian = jordan_wigner(FermionicHamiltonian)


        if threshold:
            from openfermion.ops import QubitOperator
            reduced_QubitHamiltonian = QubitOperator()
            for key in QubitHamiltonian.terms:
                if np.abs(QubitHamiltonian.terms[key]) > threshold:
                    reduced_QubitHamiltonian += QubitOperator(key, QubitHamiltonian.terms[key])
            return reduced_QubitHamiltonian
        else:
            return QubitHamiltonian


    def Get_Qubit_Hamiltonian_BK(self, threshold):
        """
        Returns the second quantised Qubit Molecular Hamiltonian of the Molecular Hamiltonian using the
        BRAVYI KITAEV transformation. First gets the fermionic Hamiltonian and than performs BK.

        e.g. H = h0 I + h1 Z0 + h2 Z1 +h3 Z2 + h4 Z3 + h5 Z0Z1 ... etc etc
        note can get integrals (h_ia and h_ijab) from Get_CCSD_Amplitudes method of Hamiltonian class!

        args:
            threshold (optional, float): gives threshold of terms to ignore... e.g. the term
                                        (0.00003+0j) [Y0 X1 X2 Y3]] would be ignored if threshold = 1e-2

        returns:
            QubitHamiltonian (openfermion.ops._qubit_operator.QubitOperator): Qubit Operator

        e.g. for H2:
                (-0.09706626861762581+0j) [] +
                (0.045302615508689394+0j) [X0 Z1 X2] +
                (0.045302615508689394+0j) [X0 Z1 X2 Z3] +
                (0.045302615508689394+0j) [Y0 Z1 Y2] +
                (0.045302615508689394+0j) [Y0 Z1 Y2 Z3] +
                (0.17141282639402383+0j) [Z0] ... etc etc

        """
        from openfermion.transforms import bravyi_kitaev

        FermionicHamiltonian = self.Get_Fermionic_Hamiltonian()
        QubitHamiltonian = bravyi_kitaev(FermionicHamiltonian)

        if threshold:
            from openfermion.ops import QubitOperator
            reduced_QubitHamiltonian = QubitOperator()
            for key in QubitHamiltonian.terms:
                if np.abs(QubitHamiltonian.terms[key]) > threshold:
                    reduced_QubitHamiltonian += QubitOperator(key, QubitHamiltonian.terms[key])
            return reduced_QubitHamiltonian
        else:
            return QubitHamiltonian

    def Get_Jordan_Wigner_CC_Matrices(self):
        """
        From list of FermionOperators (openfermion.ops._fermion_operator.FermionOperator) corresponding to
        UCCSD excitations... remember: UCCSD = ∑_pq (h_pq a†_p a_q) + ∑_pqrs (h_pqrs a†_p a†_q a_r a_s)

                    [
                       -(a†0 a2) + (a†2 a0),
                       -(a†1 a3) + (a†3 a1),
                       -(a†0 a†1 a2 a3) + a†3 a†2 a1 a0)
                    ]

        calculates corresponding sparse matrices for each term under the JORDAN WIGNER transform
        and returns them in a list. AKA returns list of matrices corresponding to each UCCSD excitation operator

        e.g. H = h0 I + h1 Z0 + h2 Z1 +h3 Z2 + h4 Z3 + h5 Z0Z1 ... etc etc
        note can get integrals (h_ia and h_ijab) from Get_CCSD_Amplitudes method of Hamiltonian class!

        e.g. output
        [
            <16x16 sparse matrix of type '<class 'numpy.complex128'>',
            <16x16 sparse matrix of type '<class 'numpy.complex128'>',
            <16x16 sparse matrix of type '<class 'numpy.complex128'>'
        ]


        returns:
            JW_CC_ops (list): List of Sparse matrices corresponding to each UCCSD excitation operator


        """

        # converts list of UCCSD fermionic operations from:  ∑_pq (h_pq a†_p a_q) + ∑_pqrs (h_pqrs a†_p a†_q a_r a_s)
        # list of matrix operations to perform each fermionic operation (under JW transform).

        #     [
        #        -(a†0 a2) + (a†2 a0),
        #        -(a†1 a3) + (a†3 a1),
        #        -(a†0 a†1 a2 a3) + a†3 a†2 a1 a0)
        #     ]

        # becomes [matrix1, matrix2, matrix3]

        from openfermion.transforms import get_sparse_operator
        JW_CC_ops = []
        for classical_op in self.Sec_Quant_CC_ops:
            # matrix operator of coupled cluster operations
            JW_CC_ops.append(get_sparse_operator(classical_op, n_qubits=self.n_qubits))
        return JW_CC_ops

    def Get_BK_CC_Matrices(self):
        """
        From list of FermionOperators (openfermion.ops._fermion_operator.FermionOperator) corresponding to
        UCCSD excitations... remember: UCCSD = ∑_pq (h_pq a†_p a_q) + ∑_pqrs (h_pqrs a†_p a†_q a_r a_s)

                    [
                       -(a†0 a2) + (a†2 a0),
                       -(a†1 a3) + (a†3 a1),
                       -(a†0 a†1 a2 a3) + a†3 a†2 a1 a0)
                    ]

        calculates corresponding sparse matrices for each term under the BK transform
        and returns them in a list. AKA returns list of matrices corresponding to each UCCSD excitation operator

        e.g. H = h0 I + h1 Z0 + h2 Z1 +h3 Z2 + h4 Z3 + h5 Z0Z1 ... etc etc
        note can get integrals (h_ia and h_ijab) from Get_CCSD_Amplitudes method of Hamiltonian class!

        e.g. output
        [
            <16x16 sparse matrix of type '<class 'numpy.complex128'>',
            <16x16 sparse matrix of type '<class 'numpy.complex128'>',
            <16x16 sparse matrix of type '<class 'numpy.complex128'>'
        ]


        returns:
            BK_CC_ops (list): List of Sparse matrices corresponding to each UCCSD excitation operator


        """

        from openfermion.transforms._bravyi_kitaev import _bravyi_kitaev_fermion_operator
        from openfermion.transforms import get_sparse_operator

        BK_CC_ops = []
        for classical_op in self.Sec_Quant_CC_ops:
            BK_Op = _bravyi_kitaev_fermion_operator(classical_op, self.n_qubits)
            BK_CC_ops.append(get_sparse_operator(BK_Op, n_qubits=self.n_qubits))
        return BK_CC_ops


class CalcEnergy():
    """
    The CalcEnergy object calculates Energies using LINEAR ALG.

    Args:
        MolecularHamiltonianMatrix (scipy.sparse.csc.csc_matrix): Sparse matrix of Molecular Hamiltonian
        n_qubits (int): Number of qubits
        JW_CC_ops (list): List of Sparse matrices corresponding to each UCCSD excitation operator
        reference_ket (scipy.sparse.csr.csr_matrix): Sparse matrix of KET corresponding to occ no basis state under
                                                     JW transform (see Get_Basis_state_in_occ_num_basis function
                                                     of Hamiltonian class)
    """
    def __init__(self, MolecularHamiltonianMatrix, reference_ket, n_qubits, JW_CC_ops_list):
        self.MolecularHamiltonianMatrix = MolecularHamiltonianMatrix
        self.reference_ket = reference_ket

        self.n_qubits = n_qubits
        self.JW_CC_ops_list = JW_CC_ops_list

    def Calc_HF_Energy(self):
        """
        Returns Hartree-Fock Energy calculated by multiplying HF state with Molecular Hamiltonian matrix.

        aka H |HF> = E_hartree |HF>

        returns:
            HF_energy (float): Hartree Fock Energy
        """

        HF_ket = self.MolecularHamiltonianMatrix.dot(self.reference_ket).toarray()  # H |HF_ref> =   E*|HF> (all in one vecotr)
        HF_energy = np.dot(np.transpose(self.reference_ket.toarray()), HF_ket)  # selects correct entries as in vector giving E (aka uses E |state vec>)
        print('HF Energy from lin alg: ', HF_energy)
        return HF_energy

    def Calc_UCCSD_No_Trot(self, parameters):
        """
        Returns UCCSD Energy calculated by applying UCCSD exponentiated matrix to HF state.
        NO TROTTERISATION PERFORMED!!!

        e.g. for H2 UCCSD_OP = exp [ t02(a†2a0−a†0a2) + t13(a†3a1−a†1a3) + t0123(a†3a†2a1a0−a†0a†1a2a3) ]

        aka:
            UCCSD_OP |HF> = |UCCSD>
        THEN:
            H |UCCSD> =  E_UCCSD |UCCSD>

        returns:
            UCCSD Energy (No trotterisation)
        """

        generator = scipy.sparse.csc_matrix((2 ** (self.n_qubits), 2 ** (self.n_qubits)), dtype=complex)
        for mat_op in range(0, len(self.JW_CC_ops_list)):
            generator = generator + parameters[mat_op] * self.JW_CC_ops_list[mat_op]
        new_state = scipy.sparse.linalg.expm_multiply(generator, self.reference_ket)
        new_bra = new_state.transpose().conj()
        assert (new_bra.dot(new_state).toarray()[0][0] - 1 < 0.0000001)
        energy = new_bra.dot(self.MolecularHamiltonianMatrix.dot(new_state))
        #print('UCCSD WITHOUT trotterisation E: ', energy.toarray()[0][0].real)
        return energy.toarray()[0][0].real

    def Calc_UCCSD_with_Trot(self, parameters):

        """
        Returns UCCSD Energy calculated by applying FIRST ORDER UCCSD operators to HF state.

        e.g.
            for H2

            UCCSD_OP = exp [ t02(a†2a0−a†0a2) + t13(a†3a1−a†1a3) + t0123(a†3a†2a1a0−a†0a†1a2a3) ]

            becomes

            UCCSD_OP^TROTTER =exp [t02(a†2a0−a†0a2)] × exp [t13(a†3a1−a†1a3)] × exp [t0123(a†3a†2a1a0−a†0a†1a2a3)]

         Function recersively applies UCCSD_OP^TROTTER to reference state (at beginning will be |HF> state)
         yielding |UCCSD_trotter> state

         finally performs
            H |UCCSD_trotter> =  E_UCCSD_trotter |UCCSD_trotter>

        returns:
            UCCSD Energy (calculated WITH trotterisation)
        """


        new_state = self.reference_ket
        for k in reversed(range(0, len(parameters))):
            new_state = scipy.sparse.linalg.expm_multiply((parameters[k] * self.JW_CC_ops_list[k]), new_state)
        new_bra = new_state.transpose().conj()
        assert (new_bra.dot(new_state).toarray()[0][0] - 1 < 0.0000001)
        energy = new_bra.dot(self.MolecularHamiltonianMatrix.dot(new_state))
        #print('UCCSD with trotterisation E: ', energy.toarray()[0][0].real)
        return energy.toarray()[0][0].real

if __name__ == '__main__':

    ### Variable Parameters
    Molecule = 'H2'
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.74))]
    n_electrons = 2
    num_shots = 10000
    ####

    ### Get Hamiltonian
    Hamilt = Hamiltonian(Molecule,
                         run_scf=1, run_mp2=1, run_cisd=1, run_ccsd=1, run_fci=1,
                         basis='sto-3g',
                         multiplicity=1,
                         geometry=geometry)  # normally None!

    Hamilt.Get_Molecular_Hamiltonian(Get_H_matrix=True)
    SQ_CC_ops, THETA_params = Hamilt.Get_ia_and_ijab_terms(Coupled_cluser_param=True)
    print(SQ_CC_ops)

    HF_transformations = Hamiltonian_Transforms(Hamilt.MolecularHamiltonian, SQ_CC_ops, Hamilt.molecule.n_qubits)

    QubitHam = HF_transformations.Get_Qubit_Hamiltonian_JW()
    print('Qubit Hamiltonian: ', QubitHam)
    QubitHam_PauliStr = HF_transformations.Convert_QubitMolecularHamiltonian_To_Pauliword_Str_list(QubitHam, Hamilt.molecule.n_qubits)
    print('Qubit Hamiltonian P_Strings: ', QubitHam_PauliStr)

    UCC_JW_excitation_matrix_list = HF_transformations.Get_Jordan_Wigner_CC_Matrices()

    HF_ref_ket, HF_ref_bra = Hamilt.Get_Basis_state_in_occ_num_basis(occupied_orbitals_index_list=[0,1]) # (|HF> , <HF|)

    # Hatree Fock Energy
    # H|HF> = E_hatree |HF>
    H_HatreeState = Hamilt.MolecularHamiltonianMatrix.dot(HF_ref_ket).toarray() #E*|HF> (all in one vecotr)
    HF_energy = np.dot(HF_ref_bra.toarray(), H_HatreeState)  #selects correct entries as in vector (look at H_HatreeState)  Energy!
    print('HF Energy from lin alg: ', HF_energy)


    w = CalcEnergy(Hamilt.MolecularHamiltonianMatrix, HF_ref_ket, Hamilt.molecule.n_qubits,
                   UCC_JW_excitation_matrix_list)
    w.Calc_HF_Energy()
    THETA_params = [2.8, 2.1, 1]

    print('UCCSD withOUT trotterisation E: ',w.Calc_UCCSD_No_Trot(THETA_params))
    print('UCCSD with trotterisation E: ',w.Calc_UCCSD_with_Trot(THETA_params))

    from quchem.Scipy_Optimizer import *
    THETA_params = [1, 2, 3]
    GG = Optimizer(w.Calc_UCCSD_with_Trot, THETA_params, 'Nelder-Mead', store_values=True, display_iter_steps=True,
                   tol=1e-5,
                   display_convergence_message=True)
    GG.get_env(50)
    GG.plot_convergence()
    plt.show()








## these functions now in CalcEnergy class ##
# import scipy
# def SPE(parameters, n_qubits, reference_ket, JW_CC_ops_list, MolecularHamiltonianMatrix):
#
#     # apply UCCSD matrix WITHOUT trotterisation!
#
#     generator = scipy.sparse.csc_matrix((2**(n_qubits), 2**(n_qubits)), dtype = complex)
#     for mat_op in range(0,len(JW_CC_ops_list)):
#         generator = generator+parameters[mat_op]*JW_CC_ops_list[mat_op]
#     new_state = scipy.sparse.linalg.expm_multiply(generator, reference_ket)
#     new_bra = new_state.transpose().conj()
#     assert(new_bra.dot(new_state).toarray()[0][0]-1<0.0000001)
#     energy = new_bra.dot(MolecularHamiltonianMatrix.dot(new_state))
#     return energy.toarray()[0][0].real
#
# def Trot_SPE(parameters, reference_ket, JW_CC_ops_list, MolecularHamiltonianMatrix):
#
#     # apply UCCSD matrix WITH first order trotterisation!
#
#     new_state = reference_ket
#     for k in reversed(range(0, len(parameters))):
#         new_state = scipy.sparse.linalg.expm_multiply((parameters[k]*JW_CC_ops_list[k]), new_state)
#     new_bra = new_state.transpose().conj()
#     assert(new_bra.dot(new_state).toarray()[0][0]-1<0.0000001)
#     energy = new_bra.dot(MolecularHamiltonianMatrix.dot(new_state))
#     return energy.toarray()[0][0].real
#
# if __name__ == '__main__':
#     THETA_params = [2.8, 2.1, 1]
#     E_pure = SPE(THETA_params, Hamilt.molecule.n_qubits, HF_ref_ket, UCC_JW_excitation_matrix_list, Hamilt.MolecularHamiltonianMatrix)
#     E_trot = Trot_SPE(THETA_params, HF_ref_ket, UCC_JW_excitation_matrix_list, Hamilt.MolecularHamiltonianMatrix)
#     print(E_pure, E_trot)















