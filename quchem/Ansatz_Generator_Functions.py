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

        Sec_Quant_CC_ia_ops = [] # second quantised single e- CC operators
        theta_parameters_ia = []
        Sec_Quant_CC_ijab_ops =[] # second quantised two e- CC operators
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

                        Sec_Quant_CC_ia_ops.append(one_elec)
                else:
                    # NO filtering
                    one_elec = FermionOperator(((a, 1), (i, 0))) - FermionOperator(((i, 1), (a, 0)))
                    if single_cc_amplitudes is not None:
                        theta_parameters_ia.append(single_cc_amplitudes[a][i])
                    else:
                        theta_parameters_ia.append(0)

                    Sec_Quant_CC_ia_ops.append(one_elec)

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

                        Sec_Quant_CC_ia_ops.append(one_elec)
                else:
                    # NO filtering
                    one_elec = FermionOperator(((a, 1), (i, 0))) - FermionOperator(((i, 1), (a, 0)))
                    if single_cc_amplitudes is not None:
                        theta_parameters_ia.append(single_cc_amplitudes[a][i])
                    else:
                        theta_parameters_ia.append(0)

                    Sec_Quant_CC_ia_ops.append(one_elec)

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
                                    theta_parameters_ijab.append(double_cc_amplitudes[a][i][b][j])
                                else:
                                    theta_parameters_ijab.append(0)
                            Sec_Quant_CC_ijab_ops.append(two_elec)
                        else:
                            # NO filtering
                            two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
                                       FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))

                            if double_cc_amplitudes is not None:
                                theta_parameters_ijab.append(double_cc_amplitudes[b][a][j][i])
                            else:
                                theta_parameters_ijab.append(0)

                            Sec_Quant_CC_ijab_ops.append(two_elec)

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
                                    theta_parameters_ijab.append(double_cc_amplitudes[a][i][b][j])
                                else:
                                    theta_parameters_ijab.append(0)
                            Sec_Quant_CC_ijab_ops.append(two_elec)
                        else:
                            # NO filtering
                            two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
                                       FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))

                            if double_cc_amplitudes is not None:
                                theta_parameters_ijab.append(double_cc_amplitudes[a][i][b][j])
                            else:
                                theta_parameters_ijab.append(0)

                            Sec_Quant_CC_ijab_ops.append(two_elec)

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
                                    theta_parameters_ijab.append(double_cc_amplitudes[a][i][b][j])
                                else:
                                    theta_parameters_ijab.append(0)
                            Sec_Quant_CC_ijab_ops.append(two_elec)
                        else:
                            # NO filtering
                            two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
                                       FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))

                            if double_cc_amplitudes is not None:
                                theta_parameters_ijab.append(double_cc_amplitudes[a][i][b][j])
                            else:
                                theta_parameters_ijab.append(0)

                            Sec_Quant_CC_ijab_ops.append(two_elec)

        return Sec_Quant_CC_ia_ops, Sec_Quant_CC_ijab_ops, theta_parameters_ia, theta_parameters_ijab

    def Remove_NOON_terms(self, NOON=None, occ_threshold=1.99999, unocc_threshold=1e-5,
                          indices_to_remove_list_manual=None, single_cc_amplitudes=None,double_cc_amplitudes=None,
                          singles_hamiltonian=None, doubles_hamiltonian=None, tol_filter_small_terms=None):

        if indices_to_remove_list_manual:
            indices_remove = set(indices_to_remove_list_manual)
        else:
            occupied_indices = np.where(NOON>occ_threshold)[0]
            occupied_indices = [index for i in occupied_indices for index in [i*2, i*2+1]]

            unoccupied_indices = np.where(NOON<unocc_threshold)[0]
            unoccupied_indices = [index for i in unoccupied_indices for index in [i * 2, i * 2 + 1]]

            indices_remove = set(occupied_indices)
            indices_remove.update(set(unoccupied_indices))

        Sec_Quant_CC_ia_ops, Sec_Quant_CC__ijab_ops, theta_parameters_ia, theta_parameters_ijab = self.Get_ia_and_ijab_terms(single_cc_amplitudes=single_cc_amplitudes,
                                                                                                                              double_cc_amplitudes=double_cc_amplitudes,
                                                                                                                              singles_hamiltonian=singles_hamiltonian,
                                                                                                                              doubles_hamiltonian=doubles_hamiltonian,
                                                                                                                              tol_filter_small_terms = tol_filter_small_terms)
        reduced_Sec_Quant_CC_ops_ia = []
        reduced_theta_parameters_ia=[]
        for index, excitation in enumerate(Sec_Quant_CC_ia_ops):
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

        if indices_to_remove_list_manual is None:
            return list(indices_remove), reduced_Sec_Quant_CC_ops_ia, reduced_Sec_Quant_CC_ops_ijab, reduced_theta_parameters_ia, reduced_theta_parameters_ijab
        else:
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
        return self.Convert_basis_state_to_occ_num_basis(HF_state_vec)

    def _Beta_BK_matrix_transform(self, n_orbitals=None):
        """
        Gives sparse Beta_BK_matrix matrix that transforms occupation number basis vectors of length n into the
        Bravyi-Kitaev basis.

        Based on  βn matrix in https://arxiv.org/pdf/1208.5986.pdf.

        matrix has form as defined in: J.Chem.Theory Comput,2018, 14, 5617−5630
        acts on occupation: | f_{0} f_{1} ... f_{n-1)>

        matrix built according to alg given in: J. Chem. Phys.137, 224109 (2012)
        this acts on | f_{n-1) ... f_{1} f_{0} > ... therefore has reversed form!

        Args:
            N_orbitals (int): Number of orbitals/qubits

        Returns:
            Beta_x (scipy.sparse.csr.csr_matrix): Sparse matrix

        """
        if n_orbitals is None:
            n_orbitals=self.n_orbitals

        from scipy.sparse import csr_matrix, kron

        I = csr_matrix(np.eye(2))

        Beta_x = csr_matrix([1]).reshape([1, 1])

        for x in range(int(np.ceil(np.log2(n_orbitals)))):
            Beta_x = kron(I, Beta_x)

            Beta_x = Beta_x.tolil()
            Beta_x[-1, :] = np.ones([1, Beta_x.shape[0]])
            Beta_x = Beta_x.tocsr()
        return Beta_x[Beta_x.shape[1]-n_orbitals:, Beta_x.shape[1]-n_orbitals:]

    def Get_BK_HF_state_in_OCC_basis(self):

        BK_mat_transform = self._Beta_BK_matrix_transform()

        # note occupation: |f_{0} f_{1}...  f_{n-1)>
        Hartree_Fock_JW_occ_basis_state = self.Get_JW_HF_state_in_OCC_basis()

        Hartree_Fock_JW_occ_basis_state = np.array(Hartree_Fock_JW_occ_basis_state).reshape([len(Hartree_Fock_JW_occ_basis_state),1])
        HF_state_BK_basis = BK_mat_transform.dot(Hartree_Fock_JW_occ_basis_state) % 2
        # modulo two very important!

        return HF_state_BK_basis.reshape([1, HF_state_BK_basis.shape[0]])[0] # note occupation: |b_{0} b_{1} ... b_{n-1)>

    def Get_BK_HF_state(self):

        BK_HF_occ_Basis = self.Get_BK_HF_state_in_OCC_basis()
        return self.Convert_occ_num_basis_to_basis_state(BK_HF_occ_Basis)


# class BK_Qubit_Reordering(Ansatz):
#
#     def __init__(self, BK_QubitHamiltonian,
#                  n_electrons,
#                  n_orbitals):
#         super().__init__(n_electrons, n_orbitals)
#
#         self.BK_QubitHamiltonian=BK_QubitHamiltonian
#
#     def Get_ia_and_ijab_terms_BK_ordering(self, single_cc_amplitudes=None, double_cc_amplitudes=None, singles_hamiltonian=None,
#                               doubles_hamiltonian=None, tol_filter_small_terms=None):
#         """
#         state encoded as: | ↑ ↑ ↑ ↑ ... ↓ ↓ ↓ ↓...> (M spin orbitals)
#
#         not | ↑ ↓  ↑ ↓ ↑ ↓...>
#         """
#
#
#         from openfermion.ops import FermionOperator
#
#         alph_occs = np.arange(0, self.n_electrons / 2, dtype=int)  # spin up occupied
#         beta_occs = np.arange(self.n_orbitals / 2, self.n_orbitals / 2 + self.n_electrons / 2, dtype=int)  # spin down occupied
#         alph_noccs = np.arange(self.n_electrons / 2, self.n_orbitals / 2, dtype=int)  # spin up un-occupied
#         beta_noccs = np.arange(self.n_orbitals / 2 + self.n_electrons / 2, self.n_orbitals, dtype=int)  # spin down UN-occupied
#
#         Sec_Quant_CC__ia_ops = []  # second quantised single e- CC operators
#         theta_parameters_ia = []
#         Sec_Quant_CC__ijab_ops = []  # second quantised two e- CC operators
#         theta_parameters_ijab = []
#
#         # SINGLE electron excitation: spin UP transition
#         for i in alph_occs:
#             i = int(i)
#             for a in alph_noccs:
#                 a=int(a)
#                 if tol_filter_small_terms:
#                     if abs(singles_hamiltonian[i][a]) > tol_filter_small_terms or abs(
#                             singles_hamiltonian[a][i]) > tol_filter_small_terms:
#                         one_elec = FermionOperator(((a, 1), (i, 0))) - FermionOperator(((i, 1), (a, 0)))
#                         if single_cc_amplitudes is not None:
#                             theta_parameters_ia.append(single_cc_amplitudes[a][i])
#                         else:
#                             theta_parameters_ia.append(0)
#
#                         Sec_Quant_CC__ia_ops.append(one_elec)
#                 else:
#                     # NO filtering
#                     one_elec = FermionOperator(((a, 1), (i, 0))) - FermionOperator(((i, 1), (a, 0)))
#                     if single_cc_amplitudes is not None:
#                         theta_parameters_ia.append(single_cc_amplitudes[a][i])
#                     else:
#                         theta_parameters_ia.append(0)
#
#                     Sec_Quant_CC__ia_ops.append(one_elec)
#
#         # SINGLE electron excitation: spin DOWN transition
#         for i in beta_occs:
#             i = int(i)
#             for a in beta_noccs:
#                 a = int(a)
#                 if tol_filter_small_terms:
#                     # uses Hamiltonian to ignore small terms!
#                     if abs(singles_hamiltonian[i][a]) > tol_filter_small_terms or abs(
#                             singles_hamiltonian[a][i]) > tol_filter_small_terms:
#                         one_elec = FermionOperator(((a, 1), (i, 0))) - FermionOperator(((i, 1), (a, 0)))
#                         if single_cc_amplitudes is not None:
#                             theta_parameters_ia.append(single_cc_amplitudes[a][i])
#                         else:
#                             theta_parameters_ia.append(0)
#
#                         Sec_Quant_CC__ia_ops.append(one_elec)
#                 else:
#                     # NO filtering
#                     one_elec = FermionOperator(((a, 1), (i, 0))) - FermionOperator(((i, 1), (a, 0)))
#                     if single_cc_amplitudes is not None:
#                         theta_parameters_ia.append(single_cc_amplitudes[a][i])
#                     else:
#                         theta_parameters_ia.append(0)
#
#                     Sec_Quant_CC__ia_ops.append(one_elec)
#
#         # DOUBLE excitation: UP + UP
#         for i in alph_occs:
#             i = int(i)
#             for j in [k for k in alph_occs if k > i]:
#                 j = int(j)
#                 for a in alph_noccs:
#                     a = int(a)
#                     for b in [k for k in alph_noccs if k > a]:
#                         b = int(b)
#                         if tol_filter_small_terms:
#                             # uses Hamiltonian to ignore small terms!
#                             if abs(doubles_hamiltonian[j][i][a][b]) > tol_filter_small_terms or abs(
#                                     doubles_hamiltonian[b][a][i][j]) > tol_filter_small_terms:
#                                 two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
#                                            FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))
#                                 if double_cc_amplitudes is not None:
#                                     theta_parameters_ijab.append(double_cc_amplitudes[b][a][j][i])
#                                 else:
#                                     theta_parameters_ijab.append(0)
#                             Sec_Quant_CC__ijab_ops.append(two_elec)
#                         else:
#                             # NO filtering
#                             two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
#                                        FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))
#
#                             if double_cc_amplitudes is not None:
#                                 theta_parameters_ijab.append(double_cc_amplitudes[b][a][j][i])
#                             else:
#                                 theta_parameters_ijab.append(0)
#
#                             Sec_Quant_CC__ijab_ops.append(two_elec)
#
#         # DOUBLE excitation: DOWN + DOWN
#         for i in beta_occs:
#             i = int(i)
#             for j in [k for k in beta_occs if k > i]:
#                 j = int(j)
#                 for a in beta_noccs:
#                     a = int(a)
#                     for b in [k for k in beta_noccs if k > a]:
#                         b = int(b)
#                         if tol_filter_small_terms:
#                             # uses Hamiltonian to ignore small terms!
#                             if abs(doubles_hamiltonian[j][i][a][b]) > tol_filter_small_terms or abs(
#                                     doubles_hamiltonian[b][a][i][j]) > tol_filter_small_terms:
#                                 two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
#                                            FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))
#                                 if double_cc_amplitudes is not None:
#                                     theta_parameters_ijab.append(double_cc_amplitudes[b][a][j][i])
#                                 else:
#                                     theta_parameters_ijab.append(0)
#                             Sec_Quant_CC__ijab_ops.append(two_elec)
#                         else:
#                             # NO filtering
#                             two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
#                                        FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))
#
#                             if double_cc_amplitudes is not None:
#                                 theta_parameters_ijab.append(double_cc_amplitudes[b][a][j][i])
#                             else:
#                                 theta_parameters_ijab.append(0)
#
#                             Sec_Quant_CC__ijab_ops.append(two_elec)
#
#         # DOUBLE excitation: up + DOWN
#         for i in alph_occs:
#             i = int(i)
#             for j in [k for k in beta_occs if k > i]:
#                 j = int(j)
#                 for a in alph_noccs:
#                     a = int(a)
#                     for b in [k for k in beta_noccs if k > a]:
#                         b = int(b)
#                         if tol_filter_small_terms:
#                             # uses Hamiltonian to ignore small terms!
#                             if abs(doubles_hamiltonian[j][i][a][b]) > tol_filter_small_terms or abs(
#                                     doubles_hamiltonian[b][a][i][j]) > tol_filter_small_terms:
#                                 two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
#                                            FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))
#                                 if double_cc_amplitudes is not None:
#                                     theta_parameters_ijab.append(double_cc_amplitudes[b][a][j][i])
#                                 else:
#                                     theta_parameters_ijab.append(0)
#                             Sec_Quant_CC__ijab_ops.append(two_elec)
#                         else:
#                             # NO filtering
#                             two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
#                                        FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))
#
#                             if double_cc_amplitudes is not None:
#                                 theta_parameters_ijab.append(double_cc_amplitudes[b][a][j][i])
#                             else:
#                                 theta_parameters_ijab.append(0)
#
#                             Sec_Quant_CC__ijab_ops.append(two_elec)
#
#         return Sec_Quant_CC__ia_ops, Sec_Quant_CC__ijab_ops, theta_parameters_ia, theta_parameters_ijab
#
#     def Get_Re_ordered_Hamiltonian(self):
#
#         from openfermion.ops import QubitOperator
#         new_Hamiltonian = QubitOperator()
#
#         for Op in self.BK_QubitHamiltonian:
#             for PauliWord, const in Op.terms.items():
#                 if PauliWord == ():
#                     new_Hamiltonian += Op
#                 else:
#                     QubitNo_list, PauliStr_list = zip(*PauliWord)
#
#                     # max_even_qubit = max(i for i in QubitNo_list if not i % 2)
#                     max_even_qubit = int(self.n_orbitals / 2)
#
#                     pauli_list = []
#                     for index, qubitNo in enumerate(QubitNo_list):
#                         if qubitNo % 2:
#                             new_qubitNo = int(max_even_qubit + (qubitNo / 2))
#                         else:
#                             new_qubitNo = int(qubitNo / 2)
#                         pauli_list.append('{}{}'.format(PauliStr_list[index], new_qubitNo))
#
#                     new_QubitOp = QubitOperator(' '.join(pauli_list), const)
#                     new_Hamiltonian += new_QubitOp
#         return new_Hamiltonian
#
#     def New_BK_HF_state(self):
#         """
#         Function to re-arrange  | ↑ ↓  ↑ ↓...> state too | ↑ ↑ ↑ ↑ ... ↓ ↓ ↓ ↓...>
#
#         """
#
#         BK_state = self.Get_BK_HF_state_in_OCC_basis() # in form | ↑ ↓  ↑ ↓ ↑ ↓...>
#
#         even_indices = np.arange(0,self.n_orbitals,2)
#         odd_indices = np.arange(1, self.n_orbitals, 2)
#
#         spin_up = np.take(BK_state, even_indices)
#         spin_down = np.take(BK_state, odd_indices)
#
#         return np.hstack((spin_up,spin_down)) # in form | ↑ ↑ ↑ ↑ ... ↓ ↓ ↓ ↓...>
#
#     def Get_Reordered_Hamiltonian_2_qubits_removed(self, n_spin_up_electrons):
#         """
#         If we encode a state as | ↑ ↑ ↑ ↑ ... ↓ ↓ ↓ ↓...> (M spin orbitals) and look at BK matrix:
#
#         matrix([[1., 0., 0., 0., 0., 0., 0., 0.],
#                 [1., 1., 0., 0., 0., 0., 0., 0.],
#                 [0., 0., 1., 0., 0., 0., 0., 0.],
#                 [1., 1., 1., 1., 0., 0., 0., 0.],
#                 [0., 0., 0., 0., 1., 0., 0., 0.],
#                 [0., 0., 0., 0., 1., 1., 0., 0.],
#                 [0., 0., 0., 0., 0., 0., 1., 0.],
#                 [1., 1., 1., 1., 1., 1., 1., 1.]])
#
#         can see for M/2 (number of spin up electrons) and M (number of spin up and spin down electrons)
#         the matrix is has leading 1 values.
#
#         As the electron number is conserved by the Hamiltonian, these qubits are only actedo n  by  the  identity
#         or  PauliZ operators.
#
#         we can re-place these operators by their corresponding eigenvalues(+1  for  the  identity,  +1  for Z_{M−1} if
#         the  total  number of electrons is EVEN, and −1 if total number of electrons is ODD,
#
#         +1 for Z{M/2−1} if the number of SPIN-UP electrons is EVEN and −1 if  ODD).
#         The  Hamiltonian  then  only  acts on (M−2) qubits, so two qubits can be removed from the  simulation.
#
#         """
#
#         # following approach in PHYS. REV. X8,031022 (2018)
#         # For a system of M spin-orbitals... we can arrange the orbital ssuch  that  the  firs tM/2  spin-orbitals  describe
#         # spin up states and the last M/2 spin orbitals describe spin down states.
#         # hence need to reorder Hamiltonian according to this!
#         from openfermion.ops import QubitOperator
#         new_Hamiltonian = QubitOperator()
#
#         for Op in self.BK_QubitHamiltonian:
#             for PauliWord, const in Op.terms.items():
#                 if PauliWord == ():
#                     new_Hamiltonian += Op
#                 else:
#                     QubitNo_list, PauliStr_list = zip(*PauliWord)
#
#                     QubitNo_list = list(QubitNo_list)
#                     PauliStr_list = list(PauliStr_list)
#
#                     if (self.n_orbitals / 2 - 1) in QubitNo_list:
#                         M_over_2_index = QubitNo_list.index((self.n_orbitals / 2 - 1))
#                         QubitNo_list.pop(M_over_2_index)
#                         PauliStr_list.pop(M_over_2_index)
#
#                         # if n_spin_up_electrons %2:  # only spin up electrons!
#                         #     const = const * -1
#                         #     print('correction')
#                         # else:
#                         #     const = const * 1
#
#                     if (self.n_orbitals - 1) in QubitNo_list:
#                         M_index = QubitNo_list.index((self.n_orbitals - 1))
#                         QubitNo_list.pop(M_index)
#                         PauliStr_list.pop(M_index)
#
#                         # if self.n_electrons %2: # BOTH spin-up and down electrons!
#                         #     const = const * -1
#                         #     print('TOTAL correction')
#                         # else:
#                         #     const = const * 1
#
#                     # re-numbering operations!
#                     pauli_list = ['{}{}'.format(PauliStr_list[index], qubitNo) if qubitNo<(self.n_orbitals / 2 - 1)
#                                      else '{}{}'.format(PauliStr_list[index], qubitNo-1) for index, qubitNo in enumerate(QubitNo_list)]
#                     new_QubitOp = QubitOperator(' '.join(pauli_list), const)
#                     new_Hamiltonian += new_QubitOp
#
#         return new_Hamiltonian
#
#     def BF_HF_state_REDUCED(self):
#         BK_state = self.New_BK_HF_state()
#         return np.delete([BK_state], [(self.n_orbitals / 2 - 1), (self.n_orbitals - 1)])

class BK_Qubit_Reduction(Ansatz):

    def __init__(self, BK_QubitHamiltonian,
                 n_electrons,
                 n_orbitals):
        super().__init__(n_electrons, n_orbitals)

        self.BK_QubitHamiltonian=BK_QubitHamiltonian

    def Get_Reordered_Hamiltonian_2_qubits_removed_TODO(self):
        """
        Under BK transform IF state is a multiple of 2 then one can re-arrange orbitals as spin up then spin down:
        | ↑ ↓  ↑ ↓...> to | ↑ ↑ ↑ ↑ ... ↓ ↓ ↓ ↓...> with (M spin orbitals) then look at BK matrix:

        matrix([[1., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 0., 0., 0., 0.],
                [0., 0., 0., 0., 1., 0., 0., 0.],
                [0., 0., 0., 0., 1., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 1., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1.]])

        can see for M/2 (number of spin up electrons) and M (number of spin up and spin down electrons)
        the matrix is has leading 1 values.

        As the electron number is conserved by the Hamiltonian, these qubits are only actedo n  by  the  identity
        or  PauliZ operators.

        we can re-place these operators by their corresponding eigenvalues(+1  for  the  identity,  +1  for Z_{M−1} if
        the  total  number of electrons is EVEN, and −1 if total number of electrons is ODD,

        +1 for Z{M/2−1} if the number of SPIN-UP electrons is EVEN and −1 if  ODD).
        The  Hamiltonian  then  only  acts on (M−2) qubits, so two qubits can be removed from the  simulation.

        """
        pass

    def Get_Reordered_BF_HF_state_2_qubits_removed_TODO(self):
        """
        Under BK transform IF state is a multiple of 2 then one can re-arrange orbitals as spin up then spin down:
        | ↑ ↓  ↑ ↓...> to | ↑ ↑ ↑ ↑ ... ↓ ↓ ↓ ↓...> with (M spin orbitals) then look at BK matrix:

        matrix([[1., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 0., 0., 0., 0.],
                [0., 0., 0., 0., 1., 0., 0., 0.],
                [0., 0., 0., 0., 1., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 1., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1.]])

        can see for M/2 (number of spin up electrons) and M (number of spin up and spin down electrons)
        the matrix is has leading 1 values.

        As the electron number is conserved by the Hamiltonian, these qubits are only actedo n  by  the  identity
        or  PauliZ operators.

        we can re-place these operators by their corresponding eigenvalues(+1  for  the  identity,  +1  for Z_{M−1} if
        the  total  number of electrons is EVEN, and −1 if total number of electrons is ODD,

        +1 for Z{M/2−1} if the number of SPIN-UP electrons is EVEN and −1 if  ODD).
        The  Hamiltonian  then  only  acts on (M−2) qubits, so two qubits can be removed from the  simulation.

        """
        pass

    def Get_Reordered_ia_and_ijab_terms_2_qubits_removed_TODO(self):
        """
        Under BK transform IF state is a power of 2 then one can re-arrange orbitals as spin up then spin down:
        | ↑ ↓  ↑ ↓...> to | ↑ ↑ ↑ ↑ ... ↓ ↓ ↓ ↓...> with (M spin orbitals) then look at BK matrix:

        matrix([[1., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 0., 0., 0., 0.],
                [0., 0., 0., 0., 1., 0., 0., 0.],
                [0., 0., 0., 0., 1., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 1., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1.]])

        can see for M/2 (number of spin up electrons) and M (number of spin up and spin down electrons)
        the matrix is has leading 1 values.

        As the electron number is conserved by the Hamiltonian, these qubits are only actedo n  by  the  identity
        or  PauliZ operators.

        we can re-place these operators by their corresponding eigenvalues(+1  for  the  identity,  +1  for Z_{M−1} if
        the  total  number of electrons is EVEN, and −1 if total number of electrons is ODD,

        +1 for Z{M/2−1} if the number of SPIN-UP electrons is EVEN and −1 if  ODD).
        The  Hamiltonian  then  only  acts on (M−2) qubits, so two qubits can be removed from the  simulation.

        """
        pass

    def Remove_indices_from_Hamiltonian_manual(self, list_of_qubit_indices_to_remove, list_of_correction_vals):

        from openfermion.ops import QubitOperator
        new_Hamiltonian = QubitOperator()

        for Op in self.BK_QubitHamiltonian:
            for PauliWord, const in Op.terms.items():
                if PauliWord == ():
                    new_Hamiltonian += Op
                else:
                    QubitNo_list, PauliStr_list = zip(*PauliWord)
                    QubitNo_list = np.array(QubitNo_list)
                    PauliStr_list = np.array(PauliStr_list)

                    indices_to_remove = np.where(np.isin(QubitNo_list, list_of_qubit_indices_to_remove) == True)[0]

                    #             print(indices_to_remove, QubitNo_list)
                    const_corr = [list_of_correction_vals[index] for index, i_remove in enumerate(indices_to_remove) if
                                  i_remove in QubitNo_list]
                    #             print(indices_to_remove, Op)
                    #             print('include:', const_corr)
                    if const_corr:
                        const = np.prod(const_corr) * const

                    QubitNo_list = np.delete(QubitNo_list, indices_to_remove)
                    PauliStr_list = np.delete(PauliStr_list, indices_to_remove)
                    new_pauli_word = list(zip(QubitNo_list.tolist(), PauliStr_list.tolist()))
                    new_Hamiltonian += QubitOperator(new_pauli_word, const)
        return new_Hamiltonian

    def Remove_indices_from_Hamiltonian(self, list_of_qubit_indices_to_remove):

        BK_State = self.Get_BK_HF_state_in_OCC_basis()
        from openfermion.ops import QubitOperator
        new_Hamiltonian = QubitOperator()

        for Op in self.BK_QubitHamiltonian:
            for PauliWord, const in Op.terms.items():
                if PauliWord == ():
                    new_Hamiltonian += Op
                else:
                    QubitNo_list, PauliStr_list = zip(*PauliWord)
                    QubitNo_list = np.array(QubitNo_list)
                    PauliStr_list = np.array(PauliStr_list)

                    indices_to_remove = np.where(np.isin(QubitNo_list, list_of_qubit_indices_to_remove) == True)[0]

                    # note indexing slightly different for BK state (as always whole)
                    # some op start at non-zero qubitNo (e.g. X2 Y2)
                    BK_State_being_lost= np.take(BK_State, list(set(QubitNo_list).intersection(list_of_qubit_indices_to_remove)))
                    PauliStr_corresponding = np.take(PauliStr_list, indices_to_remove)

                    for i, bit in enumerate(BK_State_being_lost):
                        if PauliStr_corresponding[i] == 'Z':
                            if int(bit) == 1:
                                const = const * -1
                            elif int(bit) == 0:
                                const = const * 1
                            else:
                                raise ValueError('input state is not binary')
                        else:
                            const=0
                        # if int(bit) ==1:
                        #     if PauliStr_corresponding[i]=='Y':
                        #         # as Y|1> = -i |0> ### BUT we only measure -1... don't see phase of i!
                        #         const = const*-1
                        #     elif PauliStr_corresponding[i]=='Z':
                        #         # as Z|1> = -1 |1>
                        #         const = const*-1

                    QubitNo_list = np.delete(QubitNo_list, indices_to_remove)
                    PauliStr_list = np.delete(PauliStr_list, indices_to_remove)
                    new_pauli_word = list(zip(QubitNo_list.tolist(), PauliStr_list.tolist()))

                    new_Hamiltonian += QubitOperator(new_pauli_word, const)
        return new_Hamiltonian

    def Re_label_Hamiltonian(self, QubitHamiltonian):

        # find unique qubit indices
        qubit_set = set()
        for Op in QubitHamiltonian:
            qubit_terms = list(Op.terms.keys())[0]
            if qubit_terms:  # gets rid of Identity term
                QubitNo_list, _ = zip(*qubit_terms)
                qubit_set.update(set(QubitNo_list))

        re_label_dict = dict(zip(qubit_set, range(len(qubit_set))))
        ## ^ dictionary to re-label qubits

        ## re-label Hamiltonian using unqiue qubit indices!
        from openfermion.ops import QubitOperator
        re_labelled_Hamiltonian = QubitOperator()

        for Op in QubitHamiltonian:
            for PauliWord, const in Op.terms.items():
                if PauliWord == ():
                    re_labelled_Hamiltonian += Op
                else:
                    QubitNo_list, PauliStr_list = zip(*PauliWord)
                    QubitNo_list_new = [re_label_dict[qubitNo] for qubitNo in QubitNo_list]

                    re_labelled_P_word = list(zip(QubitNo_list_new, PauliStr_list))
                    re_labelled_Hamiltonian += QubitOperator(re_labelled_P_word, const)

        return re_label_dict, re_labelled_Hamiltonian

    def New_BK_HF_state(self, list_of_qubit_indices_to_remove):
        """
        Function get new BK state
        """
        BK_state = self.Get_BK_HF_state_in_OCC_basis() # in form | ↑ ↓  ↑ ↓ ↑ ↓...>
        reduced_BK_state=np.delete(BK_state, list_of_qubit_indices_to_remove)

        # # note occupation: |f_{0} f_{1}...  f_{n-1)>
        # JW_state = self.Get_JW_HF_state_in_OCC_basis()
        #
        # # remove qubits with indices defined
        # reduced_JW_state = np.delete(JW_state, list_of_qubit_indices_to_remove)
        #
        # #Get BK transform
        # BK_mat_transform = self._Beta_BK_matrix_transform(n_orbitals=len(reduced_JW_state))
        #
        #
        # Hartree_Fock_JW_occ_basis_state_reduced = np.array(reduced_JW_state).reshape([len(reduced_JW_state),1])
        # HF_state_BK_basis = BK_mat_transform.dot(Hartree_Fock_JW_occ_basis_state_reduced) % 2
        # # modulo two very important!
        #
        # return HF_state_BK_basis.reshape([1, HF_state_BK_basis.shape[0]])[0] # note occupation: |b_{0} b_{1} ... b_{n-1)>

        return reduced_BK_state

    def Find_Qubits_only_acted_on_by_I_or_Z(self, qubit_operator_list):

        # qubit_operator_list is intended to be either list of ia_CC_terms or ijab_CC_terms
        # finds terms that don't change initial state!

        # Generate list of qubits
        qubits_to_remove = np.arange(0, self.n_orbitals, 1)

        for term in qubit_operator_list:
            for op in term:
                for PauliWord, const in op.terms.items():
                    qubitNos, PauliStrs = list(zip(*PauliWord))

                    # find where non I or Z terms are
                    indices_to_remove = np.where(np.isin(PauliStrs, ['X', 'Y']) == True)[0]
                    qubitNo_to_remove = np.take(qubitNos, indices_to_remove)

                    i_remove = np.where(np.isin(qubits_to_remove, qubitNo_to_remove) == True)[0]
                    qubits_to_remove = np.delete(qubits_to_remove, i_remove)

        return qubits_to_remove

    def Remove_indices_from_CC_qubit_operators(self, CC_qubit_operator_list,
                                                             list_of_qubit_indices_to_remove):
        from openfermion.ops import QubitOperator
        # list_of_correction_vals=[1,1]
        new_CC_qubit_operator_list = []
        for terms in CC_qubit_operator_list:
            new_CC_op = QubitOperator()

            for Op in terms:
                for PauliWord, const in Op.terms.items():
                    if PauliWord == ():
                        new_CC_op += Op
                    else:
                        QubitNo_list, PauliStr_list = zip(*PauliWord)
                        QubitNo_list = np.array(QubitNo_list)
                        PauliStr_list = np.array(PauliStr_list)

                        indices_to_remove = np.where(np.isin(QubitNo_list, list_of_qubit_indices_to_remove) == True)[0]

                        QubitNo_list = np.delete(QubitNo_list, indices_to_remove)
                        PauliStr_list = np.delete(PauliStr_list, indices_to_remove)
                        new_pauli_word = list(zip(QubitNo_list.tolist(), PauliStr_list.tolist()))
                        new_CC_op += QubitOperator(new_pauli_word, const)
            new_CC_qubit_operator_list.append(new_CC_op)

        return new_CC_qubit_operator_list

    def Re_label_CC_qubit_operators(self, re_label_dict, CC_qubit_operator_list):

        from openfermion.ops import QubitOperator
        re_labelled_CC_ops = []
        for terms in CC_qubit_operator_list:
            re_labelled_CC_op = QubitOperator()
            for Op in terms:
                for PauliWord, const in Op.terms.items():
                    if PauliWord == ():
                        re_labelled_CC_op += Op
                    else:
                        QubitNo_list, PauliStr_list = zip(*PauliWord)
                        QubitNo_list_new = [re_label_dict[qubitNo] for qubitNo in QubitNo_list]
                        re_labelled_P_word = list(zip(QubitNo_list_new, PauliStr_list))
                        re_labelled_CC_op += QubitOperator(re_labelled_P_word, const)

            re_labelled_CC_ops.append(re_labelled_CC_op)

        return re_labelled_CC_ops

    def Reduced_ia_ijab_terms(self, n_orbitals, n_electrons, qubits_indices_KEPT,
                          tol_filter_small_terms=None,
                          singles_hamiltonian=None,
                          doubles_hamiltonian=None,
                          double_cc_amplitudes=None,
                          single_cc_amplitudes=None):

        from openfermion.ops import FermionOperator

        orbitals_index = range(0, n_orbitals)
        alph_occs = list(set([k for k in orbitals_index if k % 2 == 0 and k < n_electrons]).intersection(
            qubits_indices_KEPT))  # spin up occupied
        beta_occs = list(set([k for k in orbitals_index if k % 2 == 1 and k < n_electrons]).intersection(
            qubits_indices_KEPT))  # spin down UN-occupied
        alph_noccs = list(set([k for k in orbitals_index if k % 2 == 0 and k >= n_electrons]).intersection(
            qubits_indices_KEPT))  # spin down occupied
        beta_noccs = list(set([k for k in orbitals_index if k % 2 == 1 and k >= n_electrons]).intersection(
            qubits_indices_KEPT))  # spin up UN-occupied

        Sec_Quant_CC_ia_ops = []  # second quantised single e- CC operators
        theta_parameters_ia = []
        Sec_Quant_CC_ijab_ops = []  # second quantised two e- CC operators
        theta_parameters_ijab = []

        # SINGLE electron excitation: spin UP transition
        for i in alph_occs:
            for a in alph_noccs:
                if tol_filter_small_terms:
                    if abs(singles_hamiltonian[i][a]) > tol_filter_small_terms or abs(
                            singles_hamiltonian[a][i]) > tol_filter_small_terms:
                        one_elec = FermionOperator(((a, 1), (i, 0))) - FermionOperator(((i, 1), (a, 0)))
                        if single_cc_amplitudes is not None:
                            theta_parameters_ia.append(single_cc_amplitudes[a][i])
                        else:
                            theta_parameters_ia.append(0)

                        Sec_Quant_CC_ia_ops.append(one_elec)
                else:
                    # NO filtering
                    one_elec = FermionOperator(((a, 1), (i, 0))) - FermionOperator(((i, 1), (a, 0)))
                    if single_cc_amplitudes is not None:
                        theta_parameters_ia.append(single_cc_amplitudes[a][i])
                    else:
                        theta_parameters_ia.append(0)

                    Sec_Quant_CC_ia_ops.append(one_elec)

        # SINGLE electron excitation: spin DOWN transition
        for i in beta_occs:
            for a in beta_noccs:
                if tol_filter_small_terms:
                    # uses Hamiltonian to ignore small terms!
                    if abs(singles_hamiltonian[i][a]) > tol_filter_small_terms or abs(
                            singles_hamiltonian[a][i]) > tol_filter_small_terms:
                        one_elec = FermionOperator(((a, 1), (i, 0))) - FermionOperator(((i, 1), (a, 0)))
                        if single_cc_amplitudes is not None:
                            theta_parameters_ia.append(single_cc_amplitudes[a][i])
                        else:
                            theta_parameters_ia.append(0)

                        Sec_Quant_CC_ia_ops.append(one_elec)
                else:
                    # NO filtering
                    one_elec = FermionOperator(((a, 1), (i, 0))) - FermionOperator(((i, 1), (a, 0)))
                    if single_cc_amplitudes is not None:
                        theta_parameters_ia.append(single_cc_amplitudes[a][i])
                    else:
                        theta_parameters_ia.append(0)

                    Sec_Quant_CC_ia_ops.append(one_elec)

        # DOUBLE excitation: UP + UP
        for i in alph_occs:
            for j in [k for k in alph_occs if k > i]:
                for a in alph_noccs:
                    for b in [k for k in alph_noccs if k > a]:

                        if tol_filter_small_terms:
                            # uses Hamiltonian to ignore small terms!
                            if abs(doubles_hamiltonian[j][i][a][b]) > tol_filter_small_terms or abs(
                                    doubles_hamiltonian[b][a][i][j]) > tol_filter_small_terms:
                                two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
                                           FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))
                                if double_cc_amplitudes is not None:
                                    theta_parameters_ijab.append(double_cc_amplitudes[a][i][b][j])
                                else:
                                    theta_parameters_ijab.append(0)
                            Sec_Quant_CC_ijab_ops.append(two_elec)
                        else:
                            # NO filtering
                            two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
                                       FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))

                            if double_cc_amplitudes is not None:
                                theta_parameters_ijab.append(double_cc_amplitudes[b][a][j][i])
                            else:
                                theta_parameters_ijab.append(0)

                            Sec_Quant_CC_ijab_ops.append(two_elec)

        # DOUBLE excitation: DOWN + DOWN
        for i in beta_occs:
            for j in [k for k in beta_occs if k > i]:
                for a in beta_noccs:
                    for b in [k for k in beta_noccs if k > a]:

                        if tol_filter_small_terms:
                            # uses Hamiltonian to ignore small terms!
                            if abs(doubles_hamiltonian[j][i][a][b]) > tol_filter_small_terms or abs(
                                    doubles_hamiltonian[b][a][i][j]) > tol_filter_small_terms:
                                two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
                                           FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))
                                if double_cc_amplitudes is not None:
                                    theta_parameters_ijab.append(double_cc_amplitudes[a][i][b][j])
                                else:
                                    theta_parameters_ijab.append(0)
                            Sec_Quant_CC_ijab_ops.append(two_elec)
                        else:
                            # NO filtering
                            two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
                                       FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))

                            if double_cc_amplitudes is not None:
                                theta_parameters_ijab.append(double_cc_amplitudes[a][i][b][j])
                            else:
                                theta_parameters_ijab.append(0)

                            Sec_Quant_CC_ijab_ops.append(two_elec)

        # DOUBLE excitation: up + DOWN
        for i in alph_occs:
            for j in [k for k in beta_occs if k > i]:
                for a in alph_noccs:
                    for b in [k for k in beta_noccs if k > a]:

                        if tol_filter_small_terms:
                            # uses Hamiltonian to ignore small terms!
                            if abs(doubles_hamiltonian[j][i][a][b]) > tol_filter_small_terms or abs(
                                    doubles_hamiltonian[b][a][i][j]) > tol_filter_small_terms:
                                two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
                                           FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))
                                if double_cc_amplitudes is not None:
                                    theta_parameters_ijab.append(double_cc_amplitudes[a][i][b][j])
                                else:
                                    theta_parameters_ijab.append(0)
                            Sec_Quant_CC_ijab_ops.append(two_elec)
                        else:
                            # NO filtering
                            two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
                                       FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))

                            if double_cc_amplitudes is not None:
                                theta_parameters_ijab.append(double_cc_amplitudes[a][i][b][j])
                            else:
                                theta_parameters_ijab.append(0)

                            Sec_Quant_CC_ijab_ops.append(two_elec)

        return Sec_Quant_CC_ia_ops, Sec_Quant_CC_ijab_ops, theta_parameters_ia, theta_parameters_ijab



from functools import reduce
from openfermion import qubit_operator_sparse
from scipy.linalg import expm
class Ansatz_lin_alg():
    """

    Build the ansatz state through linear algebra rather than quantum circuits.

    Args:
        n_qubits (int): Number of qubits

    Attributes:
        qubit_state_dict (dict): qubit state dict

    """

    def __init__(self, n_qubits):
        self.qubit_state_dict = {0: np.array([[1], [0]]),
                                 1: np.array([[0], [1]])}

        self.n_qubits = n_qubits

    def Get_reference_ket(self, qubit_state_list_occ_basis):
        """
        Takes in list of qubit states... e.g. [1,0,0] and returns corresponding ket.
        Not qubit indexing starts from left!

        [1,0,0] gives:

       array( [[0],
               [1],
               [0],
               [0],
               [0],
               [0],
               [0],
               [0]])

        Args:
            qubit_state_list (list): list of orbital indices that are OCCUPIED

        returns:
            reference_ket (np.array): KET of corresponding  occ no basis state
        """
        state = np.asarray(qubit_state_list_occ_basis[::-1], dtype=int)
        reference_ket = reduce(np.kron, [self.qubit_state_dict[bit] for bit in state])
        return reference_ket

    def Get_ia_UCC_matrix_NO_TROT(self, theta_ia, UCCSD_ops_ia):
        """
        operator = exp [ 𝑡02(𝑎†2𝑎0−𝑎†0𝑎2)+𝑡13(𝑎†3𝑎1−𝑎†1𝑎3)+𝑡0123(𝑎†3𝑎†2𝑎1𝑎0−𝑎†0𝑎†1𝑎2𝑎) ]
        """
        generator = scipy.sparse.csc_matrix((2 ** (self.n_qubits), 2 ** (self.n_qubits)), dtype=complex)
        for index_ia, qubit_op_ia in enumerate(UCCSD_ops_ia):
            generator += qubit_operator_sparse(qubit_op_ia, n_qubits=self.n_qubits) * theta_ia[index_ia]
        UCC_ia_operator = expm(generator)
        return UCC_ia_operator

    def Get_ijab_UCC_matrix_NO_TROT(self, theta_ijab, UCCSD_ops_ijab):
        """
        operator = exp [ 𝑡02(𝑎†2𝑎0−𝑎†0𝑎2)+𝑡13(𝑎†3𝑎1−𝑎†1𝑎3)+𝑡0123(𝑎†3𝑎†2𝑎1𝑎0−𝑎†0𝑎†1𝑎2𝑎) ]
        """
        generator = scipy.sparse.csc_matrix((2 ** (self.n_qubits), 2 ** (self.n_qubits)), dtype=complex)

        for index_ijab, qubit_op_ijab in enumerate(UCCSD_ops_ijab):
            generator += qubit_operator_sparse(qubit_op_ijab, n_qubits=self.n_qubits) * theta_ijab[index_ijab]
        UCC_ijab_operator = expm(generator)
        return UCC_ijab_operator

    def Get_ia_UCC_matrix_WITH_trot_SINGLE_STEP(self, theta_ia, UCCSD_ops_ia):
        """
        operator = exp [ 𝑡02(𝑎†2𝑎0−𝑎†0𝑎2)+𝑡13(𝑎†3𝑎1−𝑎†1𝑎3)+𝑡0123(𝑎†3𝑎†2𝑎1𝑎0−𝑎†0𝑎†1𝑎2𝑎) ]
        NOW
        operator = exp [𝑡02(𝑎†2𝑎0−𝑎†0𝑎2)] × exp[ 𝑡13(𝑎†3𝑎1−𝑎†1𝑎3)] × exp[𝑡0123(𝑎†3𝑎†2𝑎1𝑎0−𝑎†0𝑎†1𝑎2𝑎3)]
        """
        UCC_ia_operator = np.eye(2 ** (self.n_qubits), dtype=complex)
        for index_ia, qubit_op_ia in enumerate(UCCSD_ops_ia):
            qubit_op_matrix = qubit_operator_sparse(qubit_op_ia, n_qubits=self.n_qubits)
            UCC_ia_operator *= expm(qubit_op_matrix * theta_ia[index_ia])
        return UCC_ia_operator

    def Get_ijab_UCC_matrix_WITH_trot_SINGLE_STEP(self, theta_ijab, UCCSD_ops_ijab):
        """
        operator = exp [ 𝑡02(𝑎†2𝑎0−𝑎†0𝑎2)+𝑡13(𝑎†3𝑎1−𝑎†1𝑎3)+𝑡0123(𝑎†3𝑎†2𝑎1𝑎0−𝑎†0𝑎†1𝑎2𝑎) ]
        NOW
        operator = exp [𝑡02(𝑎†2𝑎0−𝑎†0𝑎2)] × exp[ 𝑡13(𝑎†3𝑎1−𝑎†1𝑎3)] × exp[𝑡0123(𝑎†3𝑎†2𝑎1𝑎0−𝑎†0𝑎†1𝑎2𝑎3)]
        """
        UCC_ijab_operator = np.eye(2 ** (self.n_qubits), dtype=complex)
        for index_ijab, qubit_op_ijab in enumerate(UCCSD_ops_ijab):
            qubit_op_matrix = qubit_operator_sparse(qubit_op_ijab, n_qubits=self.n_qubits)
            UCC_ijab_operator *= expm(qubit_op_matrix * theta_ijab[index_ijab])
        return UCC_ijab_operator

    def Get_Qubit_Hamiltonian_matrix(self, Qubit_MolecularHamiltonian):
        return qubit_operator_sparse(Qubit_MolecularHamiltonian)

    def Calc_energy_of_state(self, final_ket, Qubit_Molecular_Hamiltonian_Matrix):
        state_bra = final_ket.transpose().conj()
        energy = state_bra.dot(Qubit_Molecular_Hamiltonian_Matrix.dot(final_ket))
        return energy[0][0]


# class Ansatz_MATRIX(Ansatz):
#     """
#
#     Build the ansatz state through linear algebra rather than quantum circuits.
#
#     Args:
#         PauliWord_str_Second_Quant_CC_JW_OP_list (list): List of Fermionic Operators (openfermion.ops._fermion_operator.FermionOperator)
#         n_electrons (int): Number of electrons
#         n_qubits (int): Number of qubits
#
#     Attributes:
#         reference_ket ():
#         UCCSD_ops_matrix_list ():
#
#     """
#     def __init__(self,  n_electrons, n_orbitals, Second_Quant_CC_Ops_ia, Second_Quant_CC_Ops_ijab):
#         super().__init__(n_electrons, n_orbitals)
#         self.Second_Quant_CC_Ops_ia= Second_Quant_CC_Ops_ia
#         self.Second_Quant_CC_Ops_ijab = Second_Quant_CC_Ops_ijab
#
#     def Get_reference_HF_ket(self, transformation='JW'):
#         """
#
#         Method to obtain basis state under JW transform of state defined in occupation number basis.
#         e.g. for H2 under the Jordan Wigner transfrom has |HF> = |0011> in occ no. basis
#         occupied_orbitals_index_list = [0,1] <- as first orbitals occupied
#
#         These outputs (|HF> and <HF|) can be used with MolecularHamiltonianMatrix!.
#
#         Args:
#             occupied_orbitals_index_list (list): list of orbital indices that are OCCUPIED
#
#         returns:
#             reference_ket (scipy.sparse.csr.csr_matrix): Sparse matrix of KET corresponding to occ no basis state under
#                                                          JW transform
#
#
#         """
#         if transformation=='JW':
#             reference_ket = scipy.sparse.csc_matrix( self.Get_JW_HF_state()).transpose()
#             # reference_bra = reference_ket.transpose().conj()
#         elif transformation=='BK':
#             reference_ket = scipy.sparse.csc_matrix(self.Get_BK_HF_state()).transpose()
#             # reference_bra = reference_ket.transpose().conj()
#         else:
#             raise ValueError('unknown transformation')
#         return reference_ket
#
#     def Calc_ansatz_state_withOUT_trot(self, ia_parameters, ijab_parameters, transformation):
#
#         UCCSD_ops_ia, UCCSD_ops_ijab = self.Get_CC_Matrices(self.Second_Quant_CC_Ops_ia, self.Second_Quant_CC_Ops_ijab,
#                                                             transformation=transformation)
#
#
#         generator = scipy.sparse.csc_matrix((2 ** (self.n_orbitals), 2 ** (self.n_orbitals)), dtype=complex)
#
#         for index_ia, mat_op_ia in enumerate(UCCSD_ops_ia):
#             generator = generator + ia_parameters[index_ia] *mat_op_ia
#
#         for index_ijab, mat_op_ijab in enumerate(UCCSD_ops_ijab):
#             generator = generator + ijab_parameters[index_ijab] *mat_op_ijab
#
#         reference_ket = self.Get_reference_HF_ket(transformation=transformation)
#
#         new_state = scipy.sparse.linalg.expm_multiply(generator, reference_ket)
#         # new_bra = new_state.transpose().conj()
#         return new_state
#
#     def Calc_ansatz_state_WITH_trot_SINGLE_STEP(self, ia_parameters, ijab_parameters, transformation):
#
#         Second_Quant_CC_single_Trot_list_ia, Second_Quant_CC_single_Trot_list_ijab = self.Get_CC_Matrices(self.Second_Quant_CC_Ops_ia, self.Second_Quant_CC_Ops_ijab,
#                                                             transformation=transformation)
#
#         new_state = self.Get_reference_HF_ket(transformation=transformation)
#
#         for index_ia, mat_op_ia in enumerate(Second_Quant_CC_single_Trot_list_ia):
#             new_state = scipy.sparse.linalg.expm_multiply((ia_parameters[index_ia] * mat_op_ia), new_state)
#
#         for index_ijab, mat_op_ijab in enumerate(Second_Quant_CC_single_Trot_list_ijab):
#             new_state = scipy.sparse.linalg.expm_multiply((ijab_parameters[index_ijab] * mat_op_ijab), new_state)
#
#         return new_state
#
#     def Calc_energy_of_state(self, state_ket, Qubit_MolecularHamiltonianMatrix):
#         state_bra = state_ket.transpose().conj()
#         energy = state_bra.dot(Qubit_MolecularHamiltonianMatrix.dot(state_ket))
#         return energy.toarray()[0][0].real
#
# if __name__ == '__main__':
# #     ####### Matrix Method ######
#     from quchem.Hamiltonian_Generator_Functions import *
#     ### Parameters
#     Molecule = 'H2'#'LiH' #'H2
#     geometry = None #[('Li', (0., 0., 0.)), ('H', (0., 0., 1.45))] # [('H', (0., 0., 0.)), ('H', (0., 0., 0.74))]
#     basis = 'sto-3g'
#
#     ### Get Hamiltonian
#     Hamilt = Hamiltonian(Molecule,
#                          run_scf=1, run_mp2=1, run_cisd=1, run_ccsd=1, run_fci=1,
#                          basis=basis,
#                          multiplicity=1,
#                          geometry=geometry)  # normally None!
#
#     Hamilt.Get_Molecular_Hamiltonian(Get_H_matrix=False)
#     QubitHam = Hamilt.Get_Qubit_Hamiltonian(transformation='JW')
#     Ham_matrix_JW = Hamilt.Get_sparse_Qubit_Hamiltonian_matrix(QubitHam)
#
#     ansatz_obj = Ansatz(Hamilt.molecule.n_electrons, Hamilt.molecule.n_qubits)
#
#     Sec_Quant_CC_ia_ops, Sec_Quant_CC_ijab_ops, theta_parameters_ia, theta_parameters_ijab = ansatz_obj.Get_ia_and_ijab_terms()
#
#     ansatz_lin_alg_obj = Ansatz_MATRIX(Hamilt.molecule.n_electrons, Hamilt.molecule.n_qubits, Sec_Quant_CC_ia_ops, Sec_Quant_CC_ijab_ops)
#
#     state_ket = ansatz_lin_alg_obj.Calc_ansatz_state_withOUT_trot(theta_parameters_ia, theta_parameters_ijab, 'JW')
#     # state_ket = ansatz_lin_alg_obj.Calc_ansatz_state_WITH_trot_SINGLE_STEP(theta_parameters_ia, theta_parameters_ijab, 'JW')
#
#     Energy = ansatz_lin_alg_obj.Calc_energy_of_state(state_ket, Ham_matrix_JW)
#
#     print(Energy)
#
#     def GIVE_ENERGY(theta_ia_theta_jab_list):
#         theta_ia = theta_ia_theta_jab_list[:len(theta_parameters_ia)]
#         theta_ijab = theta_ia_theta_jab_list[len(theta_parameters_ia):]
#         state_ket = ansatz_lin_alg_obj.Calc_ansatz_state_withOUT_trot(theta_ia,
#                                                                       theta_ijab, 'JW')
#
#         # state_ket = ansatz_lin_alg_obj.Calc_ansatz_state_WITH_trot_SINGLE_STEP(theta_ia,
#         #                                                                        theta_ijab, 'JW')
#
#         Energy = ansatz_lin_alg_obj.Calc_energy_of_state(state_ket, Ham_matrix_JW)
#         return Energy
#
#     from quchem.Scipy_Optimizer import *
#     THETA_params = [*theta_parameters_ia, *theta_parameters_ijab]
#     GG = Optimizer(GIVE_ENERGY, THETA_params, 'Nelder-Mead', store_values=True, display_iter_steps=True,
#                    tol=1e-5,
#                    display_convergence_message=True)
#     GG.get_env(50)
#     GG.plot_convergence()
#     plt.show()


from quchem.quantum_circuit_functions import *
# class Ansatz_Circuit(Ansatz):
#     """
#
#     The Ansatz_Circuit object allows Hartree Fock UCCSD Ansatz Circuit to be generated
#
#     Args:
#         PauliWord_str_Second_Quant_CC_JW_OP_list (list): List of Fermionic Operators (openfermion.ops._fermion_operator.FermionOperator)
#         n_electrons (int): Number of electrons
#         n_qubits (int): Number of qubits
#
#     Attributes:
#         HF_QCirc ():
#
#     """
#     def __init__(self, Qubit_Op_list_Second_Quant_CC_Ops_ia, Qubit_Op_list_Second_Quant_CC_Ops_ijab,
#                  n_orbitals, n_electrons, manual_HF_state=None):
#
#         super().__init__(n_electrons, n_orbitals)
#
#         self.Qubit_Op_list_Second_Quant_CC_Ops_ia = Qubit_Op_list_Second_Quant_CC_Ops_ia
#         self.Qubit_Op_list_Second_Quant_CC_Ops_ijab=Qubit_Op_list_Second_Quant_CC_Ops_ijab
#         self.n_orbitals = n_orbitals
#         self.n_electrons = n_electrons
#
#         # TODO consider changing this
#         self.manual_HF_state = manual_HF_state
#
#         self.HF_QCirc = None
#
#     def _Get_HF_Quantum_Circuit(self, transformation='JW'):
#
#         if self.manual_HF_state is not None:
#             HF_state_occ_basis = self.manual_HF_state
#         else:
#             if transformation=='JW':
#                 HF_state_occ_basis = self.Get_JW_HF_state_in_OCC_basis()
#             elif transformation=='BK':
#                 HF_state_occ_basis = self.Get_BK_HF_state_in_OCC_basis()
#             else:
#                 raise ValueError('unknown transformation')
#
#         HF_state_prep = State_Prep(HF_state_occ_basis)
#         HF_state_prep_circuit = cirq.Circuit(cirq.decompose_once(
#             (HF_state_prep(*cirq.LineQubit.range(HF_state_prep.num_qubits())))))
#         self.HF_QCirc = list(HF_state_prep_circuit.all_operations())
#
#     def _Get_UCCSD_Quantum_Circuit(self, Theta_param_list_ia, Theta_param_list_ijab):
#
#         Q_Circuit_generator_list = []
#
#         for ia_index, ia_QubitOp in enumerate(self.Qubit_Op_list_Second_Quant_CC_Ops_ia):
#             for ia_QubitOp_term in ia_QubitOp:
#                 Q_circuit_gen = full_exponentiated_PauliWord_circuit(ia_QubitOp_term, Theta_param_list_ia[ia_index])
#                 Q_circuit = cirq.Circuit(cirq.decompose_once(
#                     (Q_circuit_gen(*cirq.LineQubit.range(Q_circuit_gen.num_qubits())))))
#                 Q_Circuit_generator_list.append(Q_circuit.all_operations())
#
#         for ijab_index, iajb_QubitOp in enumerate(self.Qubit_Op_list_Second_Quant_CC_Ops_ijab):
#             for ijab_QubitOp_term in iajb_QubitOp:
#                 Q_circuit_gen = full_exponentiated_PauliWord_circuit(ijab_QubitOp_term, Theta_param_list_ijab[ijab_index])
#                 Q_circuit = cirq.Circuit(cirq.decompose_once(
#                     (Q_circuit_gen(*cirq.LineQubit.range(Q_circuit_gen.num_qubits())))))
#                 Q_Circuit_generator_list.append(Q_circuit.all_operations())
#
#         return Q_Circuit_generator_list
#
#     def Get_Full_HF_UCCSD_QC(self, Theta_param_list_ia, Theta_param_list_ijab, transformation='JW'):
#
#         if self.HF_QCirc is None:
#             self._Get_HF_Quantum_Circuit(transformation=transformation)
#
#         UCCSD_QC_List = self._Get_UCCSD_Quantum_Circuit(Theta_param_list_ia, Theta_param_list_ijab)
#
#         full_circuit = cirq.Circuit(
#             [
#                 self.HF_QCirc,
#                 *UCCSD_QC_List,
#             ]
#         )
#         return full_circuit

class Ansatz_Circuit():
    """

    The Ansatz_Circuit object allows Hartree Fock UCCSD Ansatz Circuit to be generated

    Args:
        PauliWord_str_Second_Quant_CC_JW_OP_list (list): List of Fermionic Operators (openfermion.ops._fermion_operator.FermionOperator)
        n_electrons (int): Number of electrons
        n_qubits (int): Number of qubits

    Attributes:
        HF_QCirc ():

    """
    def __init__(self, input_state=None,
                 Qubit_Op_list_Second_Quant_CC_Ops_ia=None,
                 Qubit_Op_list_Second_Quant_CC_Ops_ijab=None):

        self.Qubit_Op_list_Second_Quant_CC_Ops_ia = Qubit_Op_list_Second_Quant_CC_Ops_ia
        self.Qubit_Op_list_Second_Quant_CC_Ops_ijab=Qubit_Op_list_Second_Quant_CC_Ops_ijab

        self.input_state = input_state


    def _input_state_Q_Circuit(self):

        HF_state_prep = State_Prep(self.input_state)
        HF_state_prep_circuit = cirq.Circuit(cirq.decompose_once(
            (HF_state_prep(*cirq.LineQubit.range(HF_state_prep.num_qubits())))))

        return list(HF_state_prep_circuit.all_operations())

    def _Get_UCCSD_ia_circuit(self, Theta_param_list_ia):

        Q_Circuit_generator_list = []

        for ia_index, ia_QubitOp in enumerate(self.Qubit_Op_list_Second_Quant_CC_Ops_ia):
            for ia_QubitOp_term in ia_QubitOp:
                Q_circuit_gen = full_exponentiated_PauliWord_circuit(ia_QubitOp_term, Theta_param_list_ia[ia_index])
                Q_circuit = cirq.Circuit(cirq.decompose_once(
                    (Q_circuit_gen(*cirq.LineQubit.range(Q_circuit_gen.num_qubits())))))
                Q_Circuit_generator_list.append(Q_circuit.all_operations())
        return Q_Circuit_generator_list

    def _Get_UCCSD_ijab_circuit(self, Theta_param_list_ijab):

        Q_Circuit_generator_list = []
        for ijab_index, iajb_QubitOp in enumerate(self.Qubit_Op_list_Second_Quant_CC_Ops_ijab):
            for ijab_QubitOp_term in iajb_QubitOp:
                Q_circuit_gen = full_exponentiated_PauliWord_circuit(ijab_QubitOp_term, Theta_param_list_ijab[ijab_index])
                Q_circuit = cirq.Circuit(cirq.decompose_once(
                    (Q_circuit_gen(*cirq.LineQubit.range(Q_circuit_gen.num_qubits())))))
                Q_Circuit_generator_list.append(Q_circuit.all_operations())
        return Q_Circuit_generator_list

    def Get_Full_HF_UCCSD_QC(self, Theta_param_list_ia=None, Theta_param_list_ijab=None,
                             ia_first=True):

        if self.input_state is not None:
            input_state_Q_circuit = self._input_state_Q_Circuit()
        else:
            input_state_Q_circuit=[]

        if Theta_param_list_ia is not None:
            UCC_ia_circuit = self._Get_UCCSD_ia_circuit(Theta_param_list_ia)
        else:
            UCC_ia_circuit=[]

        if Theta_param_list_ijab is not None:
            UCC_ijab_circuit = self._Get_UCCSD_ijab_circuit(Theta_param_list_ijab)
        else:
            UCC_ijab_circuit=[]

        if ia_first is True:
            full_circuit = cirq.Circuit(
                [
                    input_state_Q_circuit,
                    *UCC_ia_circuit,
                    *UCC_ijab_circuit,
                ]
            )
        else:
            full_circuit = cirq.Circuit(
                [
                    input_state_Q_circuit,
                    *UCC_ijab_circuit,
                    *UCC_ia_circuit
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