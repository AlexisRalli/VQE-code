import scipy
class Ansatz():
    def __init__(self, n_electrons, n_orbitals):
        self.n_electrons = n_electrons
        self.n_orbitals = n_orbitals

    def Get_ia_and_ijab_terms(self, single_cc_amplitudes=None, double_cc_amplitudes=None, singles_hamiltonian=None,
                              doubles_hamiltonian=None, tol_filter_small_terms = None):
        #TODO could add MP2 param option to initialise theta with MP2 amplitudes (rather than coupled cluster only option)
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
        :param single_and_double_cc_amplitudes:

        """


        # single_amplitudes and double_amplitudes from Get_CCSD_Amplitudes Hamiltonian function!

        from openfermion.ops import FermionOperator

        orbitals_index = range(0, self.n_orbitals)
        alph_occs = [k for k in orbitals_index if k % 2 == 0 and k < self.n_electrons] # spin up occupied
        beta_occs = [k for k in orbitals_index if k % 2 == 1 and k < self.n_electrons] # spin down UN-occupied
        alph_noccs = [k for k in orbitals_index if k % 2 == 0 and k >= self.n_electrons] # spin down occupied
        beta_noccs = [k for k in orbitals_index if k % 2 == 1 and k >= self.n_electrons] # spin up UN-occupied

        Sec_Quant_CC__ia_ops = [] # second quantised single e- CC operators
        theta_parameters_ia = []
        Sec_Quant_CC__ijab_ops =[] # second quantised two e- CC operators
        theta_parameters_ijab =[]


        # SINGLE electron excitation: spin UP transition
        for i in alph_occs:
            for a in alph_noccs:
                if tol_filter_small_terms:
                    if abs(singles_hamiltonian[i][a]) > tol_filter_small_terms or abs(singles_hamiltonian[a][i]) > tol_filter_small_terms:
                        one_elec = FermionOperator(((a, 1), (i, 0))) - FermionOperator(((i, 1), (a, 0)))
                        if single_cc_amplitudes is not None:
                            theta_parameters_ia.append(single_cc_amplitudes[a][i])
                        else:
                            theta_parameters_ia.append(0)

                        Sec_Quant_CC__ia_ops.append(one_elec)
                else:
                    # NO filtering
                    one_elec = FermionOperator(((a, 1), (i, 0))) - FermionOperator(((i, 1), (a, 0)))
                    if single_cc_amplitudes is not None:
                        theta_parameters_ia.append(single_cc_amplitudes[a][i])
                    else:
                        theta_parameters_ia.append(0)

                    Sec_Quant_CC__ia_ops.append(one_elec)

        # SINGLE electron excitation: spin DOWN transition
        for i in beta_occs:
            for a in beta_noccs:
                if tol_filter_small_terms:
                    # uses Hamiltonian to ignore small terms!
                    if abs(singles_hamiltonian[i][a]) > tol_filter_small_terms or abs(singles_hamiltonian[a][i]) > tol_filter_small_terms:
                        one_elec = FermionOperator(((a, 1), (i, 0))) - FermionOperator(((i, 1), (a, 0)))
                        if single_cc_amplitudes is not None:
                            theta_parameters_ia.append(single_cc_amplitudes[a][i])
                        else:
                            theta_parameters_ia.append(0)

                        Sec_Quant_CC__ia_ops.append(one_elec)
                else:
                    # NO filtering
                    one_elec = FermionOperator(((a, 1), (i, 0))) - FermionOperator(((i, 1), (a, 0)))
                    if single_cc_amplitudes is not None:
                        theta_parameters_ia.append(single_cc_amplitudes[a][i])
                    else:
                        theta_parameters_ia.append(0)

                    Sec_Quant_CC__ia_ops.append(one_elec)

        # DOUBLE excitation: UP + UP
        for i in alph_occs:
            for j in [k for k in alph_occs if k > i]:
                for a in alph_noccs:
                    for b in [k for k in alph_noccs if k > a]:

                        if tol_filter_small_terms:
                            # uses Hamiltonian to ignore small terms!
                            if abs(doubles_hamiltonian[j][i][a][b]) > tol_filter_small_terms or abs(doubles_hamiltonian[b][a][i][j]) > tol_filter_small_terms:
                                two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
                                           FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))
                                if double_cc_amplitudes is not None:
                                    theta_parameters_ijab.append(double_cc_amplitudes[b][a][j][i])
                                else:
                                    theta_parameters_ijab.append(0)
                            Sec_Quant_CC__ijab_ops.append(two_elec)
                        else:
                            # NO filtering
                            two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
                                       FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))

                            if double_cc_amplitudes is not None:
                                theta_parameters_ijab.append(double_cc_amplitudes[b][a][j][i])
                            else:
                                theta_parameters_ijab.append(0)

                            Sec_Quant_CC__ijab_ops.append(two_elec)

        # DOUBLE excitation: DOWN + DOWN
        for i in beta_occs:
            for j in [k for k in beta_occs if k > i]:
                for a in beta_noccs:
                    for b in [k for k in beta_noccs if k > a]:

                        if tol_filter_small_terms:
                            # uses Hamiltonian to ignore small terms!
                            if abs(doubles_hamiltonian[j][i][a][b]) > tol_filter_small_terms or abs(doubles_hamiltonian[b][a][i][j]) > tol_filter_small_terms:
                                two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
                                           FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))
                                if double_cc_amplitudes is not None:
                                    theta_parameters_ijab.append(double_cc_amplitudes[b][a][j][i])
                                else:
                                    theta_parameters_ijab.append(0)
                            Sec_Quant_CC__ijab_ops.append(two_elec)
                        else:
                            # NO filtering
                            two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
                                       FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))

                            if double_cc_amplitudes is not None:
                                theta_parameters_ijab.append(double_cc_amplitudes[b][a][j][i])
                            else:
                                theta_parameters_ijab.append(0)

                            Sec_Quant_CC__ijab_ops.append(two_elec)

        # DOUBLE excitation: up + DOWN
        for i in alph_occs:
            for j in [k for k in beta_occs if k > i]:
                for a in alph_noccs:
                    for b in [k for k in beta_noccs if k > a]:

                        if tol_filter_small_terms:
                            # uses Hamiltonian to ignore small terms!
                            if abs(doubles_hamiltonian[j][i][a][b]) > tol_filter_small_terms or abs(doubles_hamiltonian[b][a][i][j]) > tol_filter_small_terms:
                                two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
                                           FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))
                                if double_cc_amplitudes is not None:
                                    theta_parameters_ijab.append(double_cc_amplitudes[b][a][j][i])
                                else:
                                    theta_parameters_ijab.append(0)
                            Sec_Quant_CC__ijab_ops.append(two_elec)
                        else:
                            # NO filtering
                            two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
                                       FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))

                            if double_cc_amplitudes is not None:
                                theta_parameters_ijab.append(double_cc_amplitudes[b][a][j][i])
                            else:
                                theta_parameters_ijab.append(0)

                            Sec_Quant_CC__ijab_ops.append(two_elec)

        return Sec_Quant_CC__ia_ops, Sec_Quant_CC__ijab_ops, theta_parameters_ia, theta_parameters_ijab

    def Remove_NOON_terms(self, NMO_basis=None, occ_threshold=0.98, unocc_threshold=1e-4,
                          indices_to_remove_list_manual=None, single_cc_amplitudes=None,double_cc_amplitudes=None,
                          singles_hamiltonian=None, doubles_hamiltonian=None, tol_filter_small_terms=None):

        if indices_to_remove_list_manual:
            indices_remove = set(indices_to_remove_list_manual)
        else:
            occupied_indices = np.where(NMO_basis>occ_threshold)[0]
            unoccupied_indices = np.where(NMO_basis<unocc_threshold)[0]
            indices_remove = set(occupied_indices)
            indices_remove.update(set(unoccupied_indices))

        Sec_Quant_CC__ia_ops, Sec_Quant_CC__ijab_ops, theta_parameters_ia, theta_parameters_ijab = self.Get_ia_and_ijab_terms(single_cc_amplitudes=single_cc_amplitudes,
                                                                                                                              double_cc_amplitudes=double_cc_amplitudes,
                                                                                                                              singles_hamiltonian=singles_hamiltonian,
                                                                                                                              doubles_hamiltonian=doubles_hamiltonian,
                                                                                                                              tol_filter_small_terms = tol_filter_small_terms)
        reduced_Sec_Quant_CC_ops_ia = []
        reduced_theta_parameters_ia=[]
        for index, excitation in enumerate(Sec_Quant_CC__ia_ops):
            #each term made up of two parts: -1.0 [2^ 3^ 10 11] + 1.0 [11^ 10^ 3 2]
            first_term, second_term = excitation.terms.items()

            qubit_indices, creation_annihilation_flags = zip(*first_term[0])

            if set(qubit_indices).isdisjoint(indices_remove):
                reduced_Sec_Quant_CC_ops_ia.append(excitation)
                reduced_theta_parameters_ia.append(theta_parameters_ia[index])
            else:
                continue

        reduced_Sec_Quant_CC_ops_ijab = []
        reduced_theta_parameters_ijab=[]
        for index, excitation in enumerate(Sec_Quant_CC__ijab_ops):
            #each term made up of two parts: -1.0 [2^ 3^ 10 11] + 1.0 [11^ 10^ 3 2]
            first_term, second_term = excitation.terms.items()

            qubit_indices, creation_annihilation_flags = zip(*first_term[0])

            if set(qubit_indices).isdisjoint(indices_remove):
                reduced_Sec_Quant_CC_ops_ijab.append(excitation)
                reduced_theta_parameters_ijab.append(theta_parameters_ijab[index])
            else:
                continue

        return reduced_Sec_Quant_CC_ops_ia, reduced_Sec_Quant_CC_ops_ijab, reduced_theta_parameters_ia, reduced_theta_parameters_ijab

    def UCCSD_single_trotter_step(self,Second_Quant_CC_Ops_ia, Second_Quant_CC_Ops_ijab, transformation='JW'):

        """
        Performs single trotter step approximation of UCCSD anstaz.
            U = exp [ t02 (a†2a0−a†0a2) + t13(a†3a1−a†1a3) +t0123 (a†3a†2a1a0−a†0a†1a2a3) ]
            becomes
            U=exp [t02(a†2a0−a†0a2)] × exp [t13(a†3a1−a†1a3)] × exp [t0123(a†3a†2a1a0−a†0a†1a2a3)]

        using the JORDAN WIGNER TRANSFORM

        Takes list of UCCSD fermionic excitation operators:

                    [
                       -(a†0 a2) + (a†2 a0),
                       -(a†1 a3) + (a†3 a1),
                       -(a†0 a†1 a2 a3) + a†3 a†2 a1 a0)
                    ]
        and returns JW transform of each term and appends it to a list yielding a list of QubitOperators
        performing UCCSD.

        [
            -0.5j [X0 Z1 Y2] + 0.5j [Y0 Z1 X2],
            -0.5j [X1 Z2 Y3] + 0.5j [Y1 Z2 X3],
            0.125j [X0 X1 X2 Y3] + 0.125j [X0 X1 Y2 X3] + -0.125j [X0 Y1 X2 X3] + 0.125j [X0 Y1 Y2 Y3] +
            -0.125j [Y0 X1 X2 X3] + 0.125j [Y0 X1 Y2 Y3] + -0.125j [Y0 Y1 X2 Y3] + -0.125j [Y0 Y1 Y2 X3]
        ]

        returns:
            Second_Quant_CC_JW_OP_list (list): List of QubitOperators (openfermion.ops._qubit_operator.QubitOperator)
                                               under JW transform. Each performs a UCCSD excitation.

        """

        Second_Quant_CC_single_Trot_list_ia = []
        Second_Quant_CC_single_Trot_list_ijab=[]

        if transformation == 'JW':
            from openfermion.transforms import jordan_wigner
            for OP in Second_Quant_CC_Ops_ia:
                JW_OP = jordan_wigner(OP)
                Second_Quant_CC_single_Trot_list_ia.append(JW_OP)

            for OP in Second_Quant_CC_Ops_ijab:
                JW_OP = jordan_wigner(OP)
                Second_Quant_CC_single_Trot_list_ijab.append(JW_OP)

        elif transformation == 'BK':
            from openfermion.transforms import bravyi_kitaev
            for OP in Second_Quant_CC_Ops_ia:
                BK_OP = bravyi_kitaev(OP)
                Second_Quant_CC_single_Trot_list_ia.append(BK_OP)

            for OP in Second_Quant_CC_Ops_ijab:
                BK_OP = bravyi_kitaev(OP)
                Second_Quant_CC_single_Trot_list_ijab.append(BK_OP)

        return Second_Quant_CC_single_Trot_list_ia, Second_Quant_CC_single_Trot_list_ijab

    def UCCSD_DOUBLE_trotter_step(self):
        # TODO
        pass

    def Get_CC_Matrices(self, Sec_Quant_CC_ops_ia, Sec_Quant_CC_ops_ijab, transformation='JW'):
        """
        From list of FermionOperators (openfermion.ops._fermion_operator.FermionOperator) corresponding to
        UCCSD excitations... remember: UCCSD = ∑_pq (h_pq a†_p a_q) + ∑_pqrs (h_pqrs a†_p a†_q a_r a_s)

                    [
                       -(a†0 a2) + (a†2 a0),
                       -(a†1 a3) + (a†3 a1),
                       -(a†0 a†1 a2 a3) + a†3 a†2 a1 a0)
                    ]

        calculates corresponding sparse matrices for each term under defined transformation e.g. JORDAN WIGNER
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
            CC_op_matrices (list): List of Sparse matrices corresponding to each UCCSD excitation operator


        """

        # converts list of UCCSD fermionic operations from:  ∑_pq (h_pq a†_p a_q) + ∑_pqrs (h_pqrs a†_p a†_q a_r a_s)
        # list of matrix operations to perform each fermionic operation .

        #     [
        #        -(a†0 a2) + (a†2 a0),
        #        -(a†1 a3) + (a†3 a1),
        #        -(a†0 a†1 a2 a3) + a†3 a†2 a1 a0)
        #     ]

        # becomes [matrix1, matrix2, matrix3]

        from openfermion.transforms import get_sparse_operator
        CC_op_matrices_ia = []
        CC_op_matrices_ijab = []

        if transformation=='JW':
            from openfermion.transforms import jordan_wigner

            for classical_op in Sec_Quant_CC_ops_ia:
                # matrix operator of coupled cluster operations
                CC_op_matrices_ia.append(get_sparse_operator(jordan_wigner(classical_op), n_qubits=self.n_orbitals))

            for classical_op in Sec_Quant_CC_ops_ijab:
                # matrix operator of coupled cluster operations
                CC_op_matrices_ijab.append(get_sparse_operator(jordan_wigner(classical_op), n_qubits=self.n_orbitals))

        elif transformation=='BK':
            from openfermion.transforms import bravyi_kitaev
            for classical_op in Sec_Quant_CC_ops_ia:
                # matrix operator of coupled cluster operations
                CC_op_matrices_ia.append(get_sparse_operator(bravyi_kitaev(classical_op), n_qubits=self.n_orbitals))

            for classical_op in Sec_Quant_CC_ops_ijab:
                # matrix operator of coupled cluster operations
                CC_op_matrices_ijab.append(get_sparse_operator(bravyi_kitaev(classical_op), n_qubits=self.n_orbitals))
        else:
            raise ValueError('unknown transformation')
        return CC_op_matrices_ia, CC_op_matrices_ijab

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
        state = state.reshape([state.shape[0],1])

        Number_Qubits = int(np.log2(int(state.shape[0])))

        state_list = []

        for _ in range(Number_Qubits):
            length = int(state.shape[0])

            position_1 = np.where(state == 1)[0]

            if position_1 < length / 2:
                # single_q = np.array([[1], [0]])
                single_q = 0
                state = state[0:int(length / 2), :]
            else:
                # single_q = np.array([[0], [1]])
                single_q = 1
                state = state[int(length / 2)::, :]
            state_list.append(single_q)
        return state_list

    def Convert_occ_num_basis_to_basis_state(self, occ_num_basis):
        from openfermion import jw_configuration_state

        if isinstance(occ_num_basis, list):
            occ_num_basis = np.array(occ_num_basis)

        occupied_orbitals_index_list = np.where(occ_num_basis==1)[0]

        return jw_configuration_state(occupied_orbitals_index_list, self.n_orbitals)

    def Get_JW_HF_state(self):
        # Note output order == np.array([*unoccupied,*occupied])
        from openfermion.utils._sparse_tools import jw_hartree_fock_state
        HF_state_vec = jw_hartree_fock_state(self.n_electrons, self.n_orbitals)
        return HF_state_vec

    def Get_JW_HF_state_in_OCC_basis(self):
        # Note output order == np.array([*unoccupied,*occupied])
        HF_state_vec = self.Get_JW_HF_state()
        HF_state_vec = HF_state_vec.reshape([2**self.n_orbitals,1])
        return self.Convert_basis_state_to_occ_num_basis(HF_state_vec)[::-1] # note change of order!

    def _Beta_BK_matrix_transform(self):
        """
        Gives sparse Beta_BK_matrix matrix that transforms occupation number basis vectors of length n into the
        Bravyi-Kitaev basis.

        Based on  βn matrix in https://arxiv.org/pdf/1208.5986.pdf.


        Args:
            N_orbitals (int): Number of orbitals/qubits

        Returns:
            Beta_x (scipy.sparse.csr.csr_matrix): Sparse matrix

        """

        from scipy.sparse import csr_matrix, kron

        I = csr_matrix(np.eye(2))

        Beta_x = csr_matrix([1]).reshape([1, 1])

        for x in range(int(np.ceil(np.log2(self.n_orbitals)))):
            Beta_x = kron(I, Beta_x)

            Beta_x = Beta_x.tolil()
            Beta_x[0, :] = np.ones([1, Beta_x.shape[0]])
            Beta_x = Beta_x.tocsr()
        return Beta_x[Beta_x.shape[1]-self.n_orbitals:, Beta_x.shape[1]-self.n_orbitals:]

    def Get_BK_HF_state_in_OCC_basis(self):

        mat_transform = self._Beta_BK_matrix_transform()

        Hartree_Fock_JW_occ_basis_state = self.Get_JW_HF_state_in_OCC_basis() # note occupation: | f_{n-1) ... f_{1} f_{0} >

        Hartree_Fock_JW_occ_basis_state = np.array(Hartree_Fock_JW_occ_basis_state).reshape([len(Hartree_Fock_JW_occ_basis_state),1])
        HF_state_BK_basis = mat_transform.dot(Hartree_Fock_JW_occ_basis_state) % 2

        return HF_state_BK_basis.reshape([1, HF_state_BK_basis.shape[0]])[0] # note occupation: | b_{n-1) ... b_{1} b_{0} >

    def Get_BK_HF_state(self):
        # zero = np.array([[1], [0]])
        # one = np.array([[0], [1]])
        #
        # BK_HF_occ_Basis = self.Get_BK_HF_state_in_OCC_basis().tolist()
        #
        # from numpy import kron
        # from functools import reduce
        # STATE=[]
        # for bit in BK_HF_occ_Basis[::-1]: # note reverse order!! (if not would have | f_{n-1) ... f_{1} f_{0} > )
        #     if int(bit)==0:
        #         STATE.append(zero)
        #     elif int(bit)==1:
        #         STATE.append(one)
        #     else:
        #         raise ValueError('invalid state bit is not zero or one but: {}'.format(bit))
        # STATE_vec = reduce(kron, STATE)
        # return STATE_vec
        # from openfermion import jw_configuration_state
        # BK_HF_occ_Basis = self.Get_BK_HF_state_in_OCC_basis()
        #
        # occupied_orbitals_index_list = np.where(BK_HF_occ_Basis==1)[0]
        # BK_state = jw_configuration_state(occupied_orbitals_index_list, self.n_orbitals)
        # return BK_state
        BK_HF_occ_Basis = self.Get_BK_HF_state_in_OCC_basis()
        return self.Convert_occ_num_basis_to_basis_state(BK_HF_occ_Basis[::-1])

if __name__ == '__main__':
#     ####### Ansatz ######
    from quchem.Hamiltonian_Generator_Functions import *
    ### Parameters
    Molecule = 'H2'#'LiH' #'H2
    geometry = None #[('Li', (0., 0., 0.)), ('H', (0., 0., 1.45))] # [('H', (0., 0., 0.)), ('H', (0., 0., 0.74))]
    basis = 'sto-3g'

    ### Get Hamiltonian
    Hamilt = Hamiltonian(Molecule,
                         run_scf=1, run_mp2=1, run_cisd=1, run_ccsd=1, run_fci=1,
                         basis=basis,
                         multiplicity=1,
                         geometry=geometry)  # normally None!

    Hamilt.Get_Molecular_Hamiltonian(Get_H_matrix=False)
    QubitHam = Hamilt.Get_Qubit_Hamiltonian(transformation='JW')
    Ham_matrix_JW = Hamilt.Get_sparse_Qubit_Hamiltonian_matrix(QubitHam)

    Hamilt.Get_CCSD_Amplitudes()

    ansatz_obj = Ansatz(Hamilt.molecule.n_electrons, Hamilt.molecule.n_qubits)



    Sec_Quant_CC_ia_ops, Sec_Quant_CC_ijab_ops, theta_parameters_ia, theta_parameters_ijab = ansatz_obj.Get_ia_and_ijab_terms(single_cc_amplitudes=Hamilt.molecule.single_cc_amplitudes,
                                                                                                                            double_cc_amplitudes=Hamilt.molecule.double_cc_amplitudes,
                                                                                                                            singles_hamiltonian=Hamilt.singles_hamiltonian,
                                                                                                                            doubles_hamiltonian=Hamilt.doubles_hamiltonian,
                                                                                                                            tol_filter_small_terms = None)

    ## REDUCTION using NOON
    NMO_basis, NOON_spins_combined = Hamilt.Get_NOON()

    Sec_Quant_CC_ia_ops_REDUCED, Sec_Quant_CC_ijab_ops_REDUCED, theta_parameters_ia_REDUCED, theta_parameters_ijab_REDUCED = ansatz_obj.Remove_NOON_terms(NMO_basis=NMO_basis, occ_threshold=0.999, unocc_threshold=1e-4,
                          indices_to_remove_list_manual=None, single_cc_amplitudes=Hamilt.molecule.single_cc_amplitudes,
                                                            double_cc_amplitudes=Hamilt.molecule.double_cc_amplitudes,
                                                            singles_hamiltonian=Hamilt.singles_hamiltonian,
                                                            doubles_hamiltonian=Hamilt.doubles_hamiltonian,
                                                            tol_filter_small_terms = None)


class Ansatz_MATRIX(Ansatz):
    """

    Build the ansatz state through linear algebra rather than quantum circuits.

    Args:
        PauliWord_str_Second_Quant_CC_JW_OP_list (list): List of Fermionic Operators (openfermion.ops._fermion_operator.FermionOperator)
        n_electrons (int): Number of electrons
        n_qubits (int): Number of qubits

    Attributes:
        reference_ket ():
        UCCSD_ops_matrix_list ():

    """
    def __init__(self,  n_electrons, n_orbitals, Second_Quant_CC_Ops_ia, Second_Quant_CC_Ops_ijab):
        super().__init__(n_electrons, n_orbitals)
        self.Second_Quant_CC_Ops_ia= Second_Quant_CC_Ops_ia
        self.Second_Quant_CC_Ops_ijab = Second_Quant_CC_Ops_ijab

    def Get_reference_HF_ket(self, transformation='JW'):
        """

        Method to obtain basis state under JW transform of state defined in occupation number basis.
        e.g. for H2 under the Jordan Wigner transfrom has |HF> = |0011> in occ no. basis
        occupied_orbitals_index_list = [0,1] <- as first orbitals occupied

        These outputs (|HF> and <HF|) can be used with MolecularHamiltonianMatrix!.

        Args:
            occupied_orbitals_index_list (list): list of orbital indices that are OCCUPIED

        returns:
            reference_ket (scipy.sparse.csr.csr_matrix): Sparse matrix of KET corresponding to occ no basis state under
                                                         JW transform


        """
        if transformation=='JW':
            reference_ket = scipy.sparse.csc_matrix( self.Get_JW_HF_state()).transpose()
            # reference_bra = reference_ket.transpose().conj()
        elif transformation=='BK':
            reference_ket = scipy.sparse.csc_matrix(self.Get_BK_HF_state()).transpose()
            # reference_bra = reference_ket.transpose().conj()
        else:
            raise ValueError('unknown transformation')
        return reference_ket

    def Calc_ansatz_state_withOUT_trot(self, ia_parameters, ijab_parameters, transformation):

        UCCSD_ops_ia, UCCSD_ops_ijab = self.Get_CC_Matrices(self.Second_Quant_CC_Ops_ia, self.Second_Quant_CC_Ops_ijab,
                                                            transformation=transformation)


        generator = scipy.sparse.csc_matrix((2 ** (self.n_orbitals), 2 ** (self.n_orbitals)), dtype=complex)

        for index_ia, mat_op_ia in enumerate(UCCSD_ops_ia):
            generator = generator + ia_parameters[index_ia] *mat_op_ia

        for index_ijab, mat_op_ijab in enumerate(UCCSD_ops_ijab):
            generator = generator + ijab_parameters[index_ijab] *mat_op_ijab

        reference_ket = self.Get_reference_HF_ket(transformation=transformation)

        new_state = scipy.sparse.linalg.expm_multiply(generator, reference_ket)
        # new_bra = new_state.transpose().conj()
        return new_state

    def Calc_ansatz_state_WITH_trot_SINGLE_STEP(self, ia_parameters, ijab_parameters, transformation):

        Second_Quant_CC_single_Trot_list_ia, Second_Quant_CC_single_Trot_list_ijab = self.Get_CC_Matrices(self.Second_Quant_CC_Ops_ia, self.Second_Quant_CC_Ops_ijab,
                                                            transformation=transformation)

        new_state = self.Get_reference_HF_ket(transformation=transformation)

        for index_ia, mat_op_ia in enumerate(Second_Quant_CC_single_Trot_list_ia):
            new_state = scipy.sparse.linalg.expm_multiply((ia_parameters[index_ia] * mat_op_ia), new_state)

        for index_ijab, mat_op_ijab in enumerate(Second_Quant_CC_single_Trot_list_ijab):
            new_state = scipy.sparse.linalg.expm_multiply((ijab_parameters[index_ijab] * mat_op_ijab), new_state)

        return new_state

    def Calc_energy_of_state(self, state_ket, Qubit_MolecularHamiltonianMatrix):
        state_bra = state_ket.transpose().conj()
        energy = state_bra.dot(Qubit_MolecularHamiltonianMatrix.dot(state_ket))
        return energy.toarray()[0][0].real

if __name__ == '__main__':
#     ####### Matrix Method ######
    from quchem.Hamiltonian_Generator_Functions import *
    ### Parameters
    Molecule = 'H2'#'LiH' #'H2
    geometry = None #[('Li', (0., 0., 0.)), ('H', (0., 0., 1.45))] # [('H', (0., 0., 0.)), ('H', (0., 0., 0.74))]
    basis = 'sto-3g'

    ### Get Hamiltonian
    Hamilt = Hamiltonian(Molecule,
                         run_scf=1, run_mp2=1, run_cisd=1, run_ccsd=1, run_fci=1,
                         basis=basis,
                         multiplicity=1,
                         geometry=geometry)  # normally None!

    Hamilt.Get_Molecular_Hamiltonian(Get_H_matrix=False)
    QubitHam = Hamilt.Get_Qubit_Hamiltonian(transformation='JW')
    Ham_matrix_JW = Hamilt.Get_sparse_Qubit_Hamiltonian_matrix(QubitHam)

    ansatz_obj = Ansatz(Hamilt.molecule.n_electrons, Hamilt.molecule.n_qubits)

    Sec_Quant_CC_ia_ops, Sec_Quant_CC_ijab_ops, theta_parameters_ia, theta_parameters_ijab = ansatz_obj.Get_ia_and_ijab_terms()

    ansatz_lin_alg_obj = Ansatz_MATRIX(Hamilt.molecule.n_electrons, Hamilt.molecule.n_qubits, Sec_Quant_CC_ia_ops, Sec_Quant_CC_ijab_ops)

    state_ket = ansatz_lin_alg_obj.Calc_ansatz_state_withOUT_trot(theta_parameters_ia, theta_parameters_ijab, 'JW')
    # state_ket = ansatz_lin_alg_obj.Calc_ansatz_state_WITH_trot_SINGLE_STEP(theta_parameters_ia, theta_parameters_ijab, 'JW')

    Energy = ansatz_lin_alg_obj.Calc_energy_of_state(state_ket, Ham_matrix_JW)

    print(Energy)

    def GIVE_ENERGY(theta_ia_theta_jab_list):
        theta_ia = theta_ia_theta_jab_list[:len(theta_parameters_ia)]
        theta_ijab = theta_ia_theta_jab_list[len(theta_parameters_ia):]
        state_ket = ansatz_lin_alg_obj.Calc_ansatz_state_withOUT_trot(theta_ia,
                                                                      theta_ijab, 'JW')

        # state_ket = ansatz_lin_alg_obj.Calc_ansatz_state_WITH_trot_SINGLE_STEP(theta_ia,
        #                                                                        theta_ijab, 'JW')

        Energy = ansatz_lin_alg_obj.Calc_energy_of_state(state_ket, Ham_matrix_JW)
        return Energy

    from quchem.Scipy_Optimizer import *
    THETA_params = [*theta_parameters_ia, *theta_parameters_ijab]
    GG = Optimizer(GIVE_ENERGY, THETA_params, 'Nelder-Mead', store_values=True, display_iter_steps=True,
                   tol=1e-5,
                   display_convergence_message=True)
    GG.get_env(50)
    GG.plot_convergence()
    plt.show()


from quchem.quantum_circuit_functions import *
class Ansatz_Circuit(Ansatz):
    """

    The Ansatz_Circuit object allows Hartree Fock UCCSD Ansatz Circuit to be generated

    Args:
        PauliWord_str_Second_Quant_CC_JW_OP_list (list): List of Fermionic Operators (openfermion.ops._fermion_operator.FermionOperator)
        n_electrons (int): Number of electrons
        n_qubits (int): Number of qubits

    Attributes:
        HF_QCirc ():

    """
    def __init__(self, Qubit_Op_list_Second_Quant_CC_Ops_ia, Qubit_Op_list_Second_Quant_CC_Ops_ijab,
                 n_orbitals, n_electrons):

        super().__init__(n_electrons, n_orbitals)

        self.Qubit_Op_list_Second_Quant_CC_Ops_ia = Qubit_Op_list_Second_Quant_CC_Ops_ia
        self.Qubit_Op_list_Second_Quant_CC_Ops_ijab=Qubit_Op_list_Second_Quant_CC_Ops_ijab
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons

        self.HF_QCirc = None

    def _Get_HF_Quantum_Circuit(self, transformation='JW'):
        if transformation=='JW':
            HF_state_occ_basis = self.Get_JW_HF_state_in_OCC_basis()[::-1] # note for Q circuit qubit_0 starts!
            # AKA get the following |q_{n-1) ... q_{1) q_{0) >
            # need to reverse | q_{0) q_{1) ... q_{n-1) > as Q circuit indexes from 0 to n-1 !
        elif transformation=='BK':
            HF_state_occ_basis = self.Get_BK_HF_state_in_OCC_basis()[::-1] # note for Q circuit qubit_0 starts!
        else:
            raise ValueError('unknown transformation')

        HF_state_prep = State_Prep(HF_state_occ_basis)
        HF_state_prep_circuit = cirq.Circuit(cirq.decompose_once(
            (HF_state_prep(*cirq.LineQubit.range(HF_state_prep.num_qubits())))))
        self.HF_QCirc = list(HF_state_prep_circuit.all_operations())

    def _Get_UCCSD_Quantum_Circuit(self, Theta_param_list_ia, Theta_param_list_ijab):

        Q_Circuit_generator_list = []

        for ia_index, ia_QubitOp in enumerate(self.Qubit_Op_list_Second_Quant_CC_Ops_ia):
            for ia_QubitOp_term in ia_QubitOp:
                Q_circuit_gen = full_exponentiated_PauliWord_circuit(ia_QubitOp_term, Theta_param_list_ia[ia_index])
                Q_circuit = cirq.Circuit(cirq.decompose_once(
                    (Q_circuit_gen(*cirq.LineQubit.range(Q_circuit_gen.num_qubits())))))
                Q_Circuit_generator_list.append(Q_circuit.all_operations())

        for ijab_index, iajb_QubitOp in enumerate(self.Qubit_Op_list_Second_Quant_CC_Ops_ijab):
            for ijab_QubitOp_term in iajb_QubitOp:
                Q_circuit_gen = full_exponentiated_PauliWord_circuit(ijab_QubitOp_term, Theta_param_list_ijab[ijab_index])
                Q_circuit = cirq.Circuit(cirq.decompose_once(
                    (Q_circuit_gen(*cirq.LineQubit.range(Q_circuit_gen.num_qubits())))))
                Q_Circuit_generator_list.append(Q_circuit.all_operations())

        return Q_Circuit_generator_list

    def Get_Full_HF_UCCSD_QC(self, Theta_param_list_ia, Theta_param_list_ijab, transformation='JW'):

        if self.HF_QCirc is None:
            self._Get_HF_Quantum_Circuit(transformation=transformation)

        UCCSD_QC_List = self._Get_UCCSD_Quantum_Circuit(Theta_param_list_ia, Theta_param_list_ijab)

        full_circuit = cirq.Circuit(
            [
                self.HF_QCirc,
                *UCCSD_QC_List,
            ]
        )
        return full_circuit

if __name__ == '__main__':
#     ####### Matrix Method ######
    from quchem.Hamiltonian_Generator_Functions import *
    ### Parameters
    Molecule = 'H2'#'LiH' #'H2
    geometry = None #[('Li', (0., 0., 0.)), ('H', (0., 0., 1.45))] # [('H', (0., 0., 0.)), ('H', (0., 0., 0.74))]
    basis = 'sto-3g'

    ### Get Hamiltonian
    Hamilt = Hamiltonian(Molecule,
                         run_scf=1, run_mp2=1, run_cisd=1, run_ccsd=1, run_fci=1,
                         basis=basis,
                         multiplicity=1,
                         geometry=geometry)  # normally None!

    Hamilt.Get_Molecular_Hamiltonian(Get_H_matrix=False)
    QubitHam = Hamilt.Get_Qubit_Hamiltonian(transformation='JW')
    Ham_matrix_JW = Hamilt.Get_sparse_Qubit_Hamiltonian_matrix(QubitHam)

    ansatz_obj = Ansatz(Hamilt.molecule.n_electrons, Hamilt.molecule.n_qubits)

    Sec_Quant_CC_ia_ops, Sec_Quant_CC_ijab_ops, theta_parameters_ia, theta_parameters_ijab = ansatz_obj.Get_ia_and_ijab_terms()

    Qubit_Op_list_Second_Quant_CC_Ops_ia, Qubit_Op_list_Second_Quant_CC_Ops_ijab = ansatz_obj.UCCSD_single_trotter_step(Sec_Quant_CC_ia_ops, Sec_Quant_CC_ijab_ops,
                                                                                                                        transformation='JW')

    full_ansatz_Q_Circ = Ansatz_Circuit(Qubit_Op_list_Second_Quant_CC_Ops_ia, Qubit_Op_list_Second_Quant_CC_Ops_ijab,
                 Hamilt.molecule.n_qubits, Hamilt.molecule.n_electrons)

    ansatz_cirq_circuit = full_ansatz_Q_Circ.Get_Full_HF_UCCSD_QC(theta_parameters_ia, theta_parameters_ijab, transformation='JW')

    print(ansatz_cirq_circuit)