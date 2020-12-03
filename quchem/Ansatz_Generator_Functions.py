import scipy
from openfermion.ops import FermionOperator
from openfermion.transforms import bravyi_kitaev
from openfermion.transforms import jordan_wigner
from openfermion.transforms import get_sparse_operator
from openfermion import jw_configuration_state
from openfermion.ops import QubitOperator
from scipy.sparse.linalg import eigs
from scipy.sparse import csc_matrix


def Get_ia_terms(n_electrons, n_orbitals, single_cc_amplitudes=None,  singles_hamiltonian=None,
                               tol_filter_small_terms = None):
    """

    Get ia excitation terms as fermionic creation and annihilation operators for UCCSD.
    ia terms are standard single excitation terms (aka only occupied -> unoccupied transitions allowed)
    (faster and only marginally less accurate.)
    #TODO can add method to get pqrs terms
    #TODO these are all non-degenerate excitations which can possibly non-zero, including nocc->nocc, occ->occ, and spin-flips.
    #TODO EXPENSIVE, but will likely  get a slightly better answer.

    Args:
        n_electrons (int): number of electrons
        n_orbitals (int): number of orbitals
        singles_hamiltonian (numpy.ndarray, optional): h_pq (n_qubits x n_qubits) matrix.
        tol_filter_small_terms (bool, optional):  Whether to filter small terms in Hamiltonian (threshold currently hardcoded)
        single_cc_amplitudes (numpy.ndarray, optional): A 2-dimension array t[a,i] for CCSD single excitation amplitudes
                                where a is virtual index and i is occupied index

    returns:
        Sec_Quant_CC_ia_ops (list): list of FermionOperators (openfermion.ops._fermion_operator.FermionOperator)
        theta_parameters (list): list of theta values (parameterisation of excitation amplitudes)

     e.g.:

     n_electrons=2
     n_orbitals=4
     Sec_Quant_CC_ops, theta_parameters = Get_ia_terms(n_electrons, n_orbitals)

     Sec_Quant_CC_ops=  [
                         -1.0[0 ^ 2] + 1.0[2 ^ 0],            # -(a†0 a2) + (a†2 a0)
                         -1.0[1 ^ 3] + 1.0[3 ^ 1],            # -(a†1 a3) + (a†3 a1)
                        ]
    theta_parameters = [0,0,0]

    """

    Sec_Quant_CC_ia_ops = []  # second quantised single e- CC operators
    theta_parameters_ia = []

    # single_amplitudes and double_amplitudes from Get_CCSD_Amplitudes Hamiltonian function!
    orbitals_index = range(0, n_orbitals)

    alph_occs = [k for k in orbitals_index if k % 2 == 0 and k < n_electrons]  # spin up occupied
    beta_occs = [k for k in orbitals_index if k % 2 == 1 and k < n_electrons]  # spin down UN-occupied
    alph_noccs = [k for k in orbitals_index if k % 2 == 0 and k >= n_electrons]  # spin down occupied
    beta_noccs = [k for k in orbitals_index if k % 2 == 1 and k >= n_electrons]  # spin up UN-occupied


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

    return Sec_Quant_CC_ia_ops, theta_parameters_ia


def Get_ijab_terms(n_electrons, n_orbitals, double_cc_amplitudes=None, doubles_hamiltonian=None,
                 tol_filter_small_terms=None):
    """

    Get ijab excitation terms as fermionic creation and annihilation operators for UCCSD.
    ijab terms are standard double excitation terms (aka only occupied -> unoccupied transitions allowed)
    (faster and only marginally less accurate.)
    #TODO can add method to get pqrs terms
    #TODO these are all non-degenerate excitations which can possibly non-zero, including nocc->nocc, occ->occ, and spin-flips.
    #TODO EXPENSIVE, but will likely  get a slightly better answer.

    Args:
        n_electrons (int): number of electrons
        n_orbitals (int): number of orbitals
        doubles_hamiltonian (numpy.ndarray, optional): h_pqrs (n_qubits x n_qubits x n_qubits x n_qubits) matrix
        tol_filter_small_terms (bool, optional):  Whether to filter small terms in Hamiltonian (threshold currently hardcoded)
        double_cc_amplitudes (numpy.ndarray, optional): A 4-dimension array t[a,i,b,j] for CCSD double excitation amplitudes
                                                        where a, b are virtual indices and i, j are occupied indices.

    returns:
        Sec_Quant_CC_ijab_ops (list): list of FermionOperators (openfermion.ops._fermion_operator.FermionOperator)
        theta_parameters (list): list of theta values (parameterisation of excitation amplitudes)

     e.g.:

     n_electrons=2
     n_orbitals=4
     Sec_Quant_CC_ops, theta_parameters = Get_ijab_terms(n_electrons, n_orbitals)

     Sec_Quant_CC_ops=  [
                            -1.0[0 ^ 1 ^ 2 3] + 1.0 [3^ 2^ 1 0]  # -(a†0 a†1 a2 a3) + a†3 a†2 a1 a0)
                        ]
    theta_parameters = [0]
    """


    # single_amplitudes and double_amplitudes from Get_CCSD_Amplitudes Hamiltonian function!
    orbitals_index = range(0, n_orbitals)

    alph_occs = [k for k in orbitals_index if k % 2 == 0 and k < n_electrons]  # spin up occupied
    beta_occs = [k for k in orbitals_index if k % 2 == 1 and k < n_electrons]  # spin down UN-occupied
    alph_noccs = [k for k in orbitals_index if k % 2 == 0 and k >= n_electrons]  # spin down occupied
    beta_noccs = [k for k in orbitals_index if k % 2 == 1 and k >= n_electrons]  # spin up UN-occupied

    Sec_Quant_CC_ijab_ops = []  # second quantised two e- CC operators
    theta_parameters_ijab = []

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

    return Sec_Quant_CC_ijab_ops, theta_parameters_ijab


def Fermi_ops_to_qubit_ops(List_Fermi_Ops, transformation='JW'):

        """
        Takes list of fermionic excitation operators:

                    [
                       -(a†0 a2) + (a†2 a0),
                       -(a†1 a3) + (a†3 a1),
                       -(a†0 a†1 a2 a3) + a†3 a†2 a1 a0)
                    ]
        and returns JW/BK transform of each term and appends it to a list yielding a list of QubitOperators

        [
            -0.5j [X0 Z1 Y2] + 0.5j [Y0 Z1 X2],
            -0.5j [X1 Z2 Y3] + 0.5j [Y1 Z2 X3],
            0.125j [X0 X1 X2 Y3] + 0.125j [X0 X1 Y2 X3] + -0.125j [X0 Y1 X2 X3] + 0.125j [X0 Y1 Y2 Y3] +
            -0.125j [Y0 X1 X2 X3] + 0.125j [Y0 X1 Y2 Y3] + -0.125j [Y0 Y1 X2 Y3] + -0.125j [Y0 Y1 Y2 X3]
        ]

        returns:
            List_Qubit_Ops (list): List of QubitOperators (openfermion.ops._qubit_operator.QubitOperator)
                                   under JW/BK transform.

        """

        List_Qubit_Ops = []

        if transformation == 'JW':
            for OP in List_Fermi_Ops:
                JW_OP = jordan_wigner(OP)
                List_Qubit_Ops.append(JW_OP)

        elif transformation == 'BK':
            for OP in List_Fermi_Ops:
                BK_OP = bravyi_kitaev(OP)
                List_Qubit_Ops.append(BK_OP)

        else:
            raise ValueError('unknown transformation: {}'.format(transformation))

        return List_Qubit_Ops


class Ansatz():
    def __init__(self, n_electrons, n_orbitals):
        self.n_electrons = n_electrons
        self.n_orbitals = n_orbitals

        self.Sec_Quant_CC_ia_Fermi_ops=None
        self.theta_ia=None
        self.Sec_Quant_CC_ijab_Fermi_ops=None
        self.theta_ijab = None

        self.NOON_Sec_Quant_CC_ia_Fermi_ops=None
        self.NOON_theta_ia=None
        self.NOON_Sec_Quant_CC_ijab_Fermi_ops=None
        self.NOON_theta_ijab = None
        self.NOON_indices_removed=None


        self.Second_Quant_CC_Ops_ia=None
        self.Second_Quant_CC_single_Trot_list_ijab = None

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
        """
        self.Sec_Quant_CC_ia_Fermi_ops, self.theta_ia = Get_ia_terms(self.n_electrons,
                                                                    self.n_orbitals,
                                                                    single_cc_amplitudes=single_cc_amplitudes,
                                                                    singles_hamiltonian=singles_hamiltonian,
                                                                    tol_filter_small_terms = tol_filter_small_terms)

        self.Sec_Quant_CC_ijab_Fermi_ops, self.theta_ijab= Get_ijab_terms(self.n_electrons,
                                                                         self.n_orbitals,
                                                                         double_cc_amplitudes=double_cc_amplitudes,
                                                                         doubles_hamiltonian=doubles_hamiltonian,
                                                                         tol_filter_small_terms=tol_filter_small_terms)


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

            if (self.Sec_Quant_CC_ia_Fermi_ops is None) or (self.Sec_Quant_CC_ijab_Fermi_ops is None):
                self.Get_ia_and_ijab_terms(single_cc_amplitudes=single_cc_amplitudes,
                                           double_cc_amplitudes=double_cc_amplitudes,
                                           singles_hamiltonian=singles_hamiltonian,
                                           doubles_hamiltonian=doubles_hamiltonian,
                                           tol_filter_small_terms = tol_filter_small_terms)

        reduced_Sec_Quant_CC_ops_ia = []
        reduced_theta_parameters_ia=[]
        for index, excitation in enumerate(self.Sec_Quant_CC_ia_Fermi_ops):
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
        for index, excitation in enumerate(self.Sec_Quant_CC_ijab_Fermi_ops):
            #each term made up of two parts: -1.0 [2^ 3^ 10 11] + 1.0 [11^ 10^ 3 2]
            first_term, second_term = excitation.terms.items()

            qubit_indices, creation_annihilation_flags = zip(*first_term[0])

            if set(qubit_indices).isdisjoint(indices_remove):
                reduced_Sec_Quant_CC_ops_ijab.append(excitation)
                reduced_theta_parameters_ijab.append(theta_parameters_ijab[index])
            else:
                continue

        if indices_to_remove_list_manual is None:
            self.NOON_indices_removed= list(indices_remove)
        else:
            self.NOON_indices_removed = indices_to_remove_list_manual

        self.NOON_Sec_Quant_CC_ia_Fermi_ops=reduced_Sec_Quant_CC_ops_ia
        self.NOON_theta_ia= reduced_theta_parameters_ia
        self.NOON_Sec_Quant_CC_ijab_Fermi_ops = reduced_Sec_Quant_CC_ops_ijab
        self.NOON_theta_ijab = reduced_theta_parameters_ijab

        if indices_to_remove_list_manual is None:
            return list(indices_remove), reduced_Sec_Quant_CC_ops_ia, reduced_Sec_Quant_CC_ops_ijab, reduced_theta_parameters_ia, reduced_theta_parameters_ijab
        else:
            return reduced_Sec_Quant_CC_ops_ia, reduced_Sec_Quant_CC_ops_ijab, reduced_theta_parameters_ia, reduced_theta_parameters_ijab


    def UCCSD_single_trotter_step(self,transformation, List_FermiOps_ia=None, List_FermiOps_ijab=None):

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
        ##ia
        if List_FermiOps_ia is None:
            if self.Sec_Quant_CC_ia_Fermi_ops is None:
                self.Get_ia_and_ijab_terms()
            self.Second_Quant_CC_single_Trot_list_ia = Fermi_ops_to_qubit_ops(self.Sec_Quant_CC_ia_Fermi_ops,
                                                                 transformation=transformation)
        else:
            self.Second_Quant_CC_single_Trot_list_ia= Fermi_ops_to_qubit_ops(List_FermiOps_ia,
                                                                transformation=transformation)

        ## ijab
        if List_FermiOps_ijab is None:
            if self.Sec_Quant_CC_ijab_Fermi_ops is None:
                self.Get_ia_and_ijab_terms()
            self.Second_Quant_CC_single_Trot_list_ijab = Fermi_ops_to_qubit_ops(self.Sec_Quant_CC_ijab_Fermi_ops,
                                                                 transformation=transformation)
        else:
            self.Second_Quant_CC_single_Trot_list_ijab = Fermi_ops_to_qubit_ops(List_FermiOps_ijab,
                                                                                transformation=transformation)


    def UCCSD_DOUBLE_trotter_step(self):
        # TODO
        pass

    def Get_CC_Matrices(self, List_ia_ops, List_ijab_ops, transformation):
        """
        From list of FermionOperators (openfermion.ops._fermion_operator.FermionOperator) corresponding to
        UCCSD excitations... remember: UCCSD = ∑_pq (h_pq a†_p a_q) + ∑_pqrs (h_pqrs a†_p a†_q a_r a_s)

                    [
                       -(a†0 a2) + (a†2 a0),
                       -(a†1 a3) + (a†3 a1),
                       -(a†0 a†1 a2 a3) + a†3 a†2 a1 a0)
                    ]

        calculates corresponding sparse matrices for each term under defined transformation e.g. JORDAN WIGNER
        and returns them in a list. AKA can returns list of matrices corresponding to each UCCSD excitation operator.
        (note will work with qubit operators too)

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


        CC_op_matrices_ia = []
        CC_op_matrices_ijab = []

        if transformation=='JW':
            for classical_op in List_ia_ops:
                CC_op_matrices_ia.append(get_sparse_operator(jordan_wigner(classical_op), n_qubits=self.n_orbitals))

            for classical_op in List_ijab_ops:
                CC_op_matrices_ijab.append(get_sparse_operator(jordan_wigner(classical_op), n_qubits=self.n_orbitals))

        elif transformation=='BK':
            for classical_op in List_ia_ops:
                # matrix operator of coupled cluster operations
                CC_op_matrices_ia.append(get_sparse_operator(bravyi_kitaev(classical_op), n_qubits=self.n_orbitals))

            for classical_op in List_ijab_ops:
                CC_op_matrices_ijab.append(get_sparse_operator(bravyi_kitaev(classical_op), n_qubits=self.n_orbitals))
        else:
            raise ValueError('unknown transformation: {}'.format(transformation))
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

        for _ in range(int(np.ceil(np.log2(n_orbitals)))):
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


def Find_I_Z_indices_in_Hamiltonian(QubitHamiltonian, N_qubits):

    # all indices
    qubit_Nos_to_remove = np.arange(0, N_qubits, 1)
    for op in QubitHamiltonian:
        for PauliWord, const in op.terms.items():
            if PauliWord:
                qubitNos, PauliStrs = list(zip(*PauliWord))

                # remove indices if X or Y present
                indices_to_remove = np.where(np.isin(PauliStrs, ['X', 'Y']) == True)[0]
                qubitNo_to_remove = np.take(qubitNos, indices_to_remove)

                i_remove = np.where(np.isin(qubit_Nos_to_remove, qubitNo_to_remove) == True)[0]
                qubit_Nos_to_remove = np.delete(qubit_Nos_to_remove, i_remove)

    return qubit_Nos_to_remove

def Remove_Z_terms_from_Hamiltonian(QubitHamiltonian, input_state, list_of_qubit_indices_to_remove,
                                    check_reduction=False):

    new_Hamiltonian = QubitOperator()

    for Op in QubitHamiltonian:
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
                bits_of_state_being_lost = np.take(input_state,
                                              list(set(QubitNo_list).intersection(list_of_qubit_indices_to_remove)))
                Pstr_of_state_being_lost = np.take(PauliStr_list, indices_to_remove)

                for i, bit in enumerate(bits_of_state_being_lost):
                    if Pstr_of_state_being_lost[i] == 'Z':
                        if int(bit) == 1:
                            const = const * -1
                        elif int(bit) == 0:
                            const = const * 1
                        else:
                            raise ValueError('input state is not binary: {}'.format())
                    elif (Pstr_of_state_being_lost[i] == 'Y') or (Pstr_of_state_being_lost[i] == 'X'):
                        const = 0
                        break
                    else:
                        raise ValueError('operation is not a Pword: {}'.format(Pstr_of_state_being_lost))


                QubitNo_list = np.delete(QubitNo_list, indices_to_remove)
                PauliStr_list = np.delete(PauliStr_list, indices_to_remove)
                new_pauli_word = list(zip(QubitNo_list.tolist(), PauliStr_list.tolist()))

                new_Hamiltonian += QubitOperator(new_pauli_word, const)

    if check_reduction:
        old_H_matrix = csc_matrix(get_sparse_operator(QubitHamiltonian))#.todense()
        N_qubits = int(np.log2(old_H_matrix.shape[0]))
        new_H_matrix = csc_matrix(get_sparse_operator(new_Hamiltonian, n_qubits=N_qubits))#.todense()

        eig_values_old, eig_vectors_old = eigs(old_H_matrix)
        FCI_old = min(eig_values_old)

        eig_values_new, eig_vectors_new = eigs(new_H_matrix)
        FCI_new = min(eig_values_new)

        if not np.isclose(FCI_old, FCI_new, atol=1e-7):
            raise ValueError('Hamiltonian reduction incorrect')

    return new_Hamiltonian

def Re_label_Hamiltonian(QubitHamiltonian):

    # find unique qubit indices
    qubit_set = set()
    for Op in QubitHamiltonian:
        qubit_terms = list(Op.terms.keys())[0]
        if qubit_terms:  # gets rid of Identity term
            QubitNo_list, _ = zip(*qubit_terms)
            qubit_set.update(set(QubitNo_list))

    qubitNo_re_label_dict = dict(zip(qubit_set, range(len(qubit_set))))
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
                QubitNo_list_new = [qubitNo_re_label_dict[qubitNo] for qubitNo in QubitNo_list]

                re_labelled_P_word = list(zip(QubitNo_list_new, PauliStr_list))
                re_labelled_Hamiltonian += QubitOperator(re_labelled_P_word, const)

    return qubitNo_re_label_dict, re_labelled_Hamiltonian

def Remove_indices_from_qubit_ops(List_QubitOperators, list_of_qubit_indices_to_remove):

    new_op_list = []

    for Op in List_QubitOperators:
        new_op =QubitOperator()
        for PauliWord, const in Op.terms.items():
            if PauliWord == ():
                new_op += Op
            else:
                QubitNo_list, PauliStr_list = zip(*PauliWord)
                QubitNo_list = np.array(QubitNo_list)
                PauliStr_list = np.array(PauliStr_list)

                indices_to_remove = np.where(np.isin(QubitNo_list, list_of_qubit_indices_to_remove) == True)[0]


                QubitNo_list = np.delete(QubitNo_list, indices_to_remove)
                PauliStr_list = np.delete(PauliStr_list, indices_to_remove)
                new_pauli_word = list(zip(QubitNo_list.tolist(), PauliStr_list.tolist()))

                new_op += QubitOperator(new_pauli_word, const)
        new_op_list.append(new_op)

    return new_op_list

def Re_label_qubit_operators(qubitNo_re_label_dict, List_QubitOperators):

    re_labelled_CC_ops = []
    for terms in List_QubitOperators:
        re_labelled_CC_op = QubitOperator()
        for Op in terms:
            for PauliWord, const in Op.terms.items():
                if PauliWord == ():
                    re_labelled_CC_op += Op
                else:
                    QubitNo_list, PauliStr_list = zip(*PauliWord)
                    QubitNo_list_new = [qubitNo_re_label_dict[qubitNo] for qubitNo in QubitNo_list]
                    re_labelled_P_word = list(zip(QubitNo_list_new, PauliStr_list))
                    re_labelled_CC_op += QubitOperator(re_labelled_P_word, const)

        re_labelled_CC_ops.append(re_labelled_CC_op)

    return re_labelled_CC_ops

def Re_label_fermion_operators(qubitNo_re_label_dict, List_FermiOperators): #TODO!!!!

    re_labelled_Fermi_ops_list = []
    for double_Fermi_op_list in List_FermiOperators:
        re_labelled_Fermi_op = FermionOperator()

        op1, op2 = double_Fermi_op_list.terms.items()

        new_op1 = tuple((qubitNo_re_label_dict[index], crea) if index in qubitNo_re_label_dict.keys() else (index, crea) for index, crea in op1[0])
        new_op2 = tuple((qubitNo_re_label_dict[index], crea) if index in qubitNo_re_label_dict.keys() else (index, crea) for
                   index, crea in op2[0])

        re_labelled_Fermi_op+= FermionOperator(new_op1, op1[1])
        re_labelled_Fermi_op += FermionOperator(new_op2, op2[1])
        re_labelled_Fermi_ops_list.append(re_labelled_Fermi_op)

    return re_labelled_Fermi_ops_list


class Ansatz_Reduction(Ansatz):

    def __init__(self, QubitHamiltonian,
                 n_electrons,
                 n_orbitals):
        super().__init__(n_electrons, n_orbitals)

        self.QubitHamiltonian=QubitHamiltonian

        self.qubitNo_re_label_dict=None
        self.reduced_QubitHamiltonian=None

    def Get_removed_I_Z_Hamiltonian_relabled(self, input_state, check_reduction=False):
        qubitNos_to_remove = Find_I_Z_indices_in_Hamiltonian(self.QubitHamiltonian, self.n_orbitals)

        Reduced_H = Remove_Z_terms_from_Hamiltonian(self.QubitHamiltonian,
                                                    input_state,
                                                    qubitNos_to_remove,
                                                    check_reduction=check_reduction)

        self.qubitNo_re_label_dict, self.reduced_QubitHamiltonian = Re_label_Hamiltonian(Reduced_H)
        self.new_input = np.take(input_state, list(self.qubitNo_re_label_dict.keys()))
        self.qubitNos_to_remove = qubitNos_to_remove


        self.new_electron_count = sum(np.take(self.Get_JW_HF_state_in_OCC_basis(), list(self.qubitNo_re_label_dict.keys())))
        self.new_orbital_count = self.new_input.shape[0]

    def Relabel_Op_list(self, Qubit_operator_list):
        # intended to re-label CC operators
        re_labelled_CC_ops = []
        for terms in Qubit_operator_list:
            re_labelled_CC_op = QubitOperator()
            for Op in terms:
                for PauliWord, const in Op.terms.items():
                    if PauliWord == ():
                        re_labelled_CC_op += Op
                    else:
                        QubitNo_list, PauliStr_list = zip(*PauliWord)
                        QubitNo_list_new = [self.qubitNo_re_label_dict[qubitNo] for qubitNo in QubitNo_list]
                        re_labelled_P_word = list(zip(QubitNo_list_new, PauliStr_list))
                        re_labelled_CC_op += QubitOperator(re_labelled_P_word, const)
            re_labelled_CC_ops.append(re_labelled_CC_op)

        return re_labelled_CC_ops


    def Reduced_ia_ijab_terms(self,
                          tol_filter_small_terms=None,
                          singles_hamiltonian=None,
                          doubles_hamiltonian=None,
                          double_cc_amplitudes=None,
                          single_cc_amplitudes=None):

        qubits_indices_KEPT = list(self.qubitNo_re_label_dict.keys())

        orbitals_index = range(0, self.n_orbitals)
        alph_occs = list(set([k for k in orbitals_index if k % 2 == 0 and k < self.n_electrons]).intersection(
            qubits_indices_KEPT))  # spin up occupied
        beta_occs = list(set([k for k in orbitals_index if k % 2 == 1 and k < self.n_electrons]).intersection(
            qubits_indices_KEPT))  # spin down UN-occupied
        alph_noccs = list(set([k for k in orbitals_index if k % 2 == 0 and k >= self.n_electrons]).intersection(
            qubits_indices_KEPT))  # spin down occupied
        beta_noccs = list(set([k for k in orbitals_index if k % 2 == 1 and k >= self.n_electrons]).intersection(
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

    def Get_reduced_ia_ijab_terms(self,
                                  tol_filter_small_terms=None,
                                  singles_hamiltonian=None,
                                  doubles_hamiltonian=None,
                                  double_cc_amplitudes=None,
                                  single_cc_amplitudes=None):


        Sec_Quant_CC_ia_ops, Sec_Quant_CC_ijab_ops, theta_parameters_ia, theta_parameters_ijab = self.Reduced_ia_ijab_terms(single_cc_amplitudes=single_cc_amplitudes,
                                   double_cc_amplitudes=double_cc_amplitudes,
                                   singles_hamiltonian=singles_hamiltonian,
                                   doubles_hamiltonian=doubles_hamiltonian,
                                   tol_filter_small_terms=tol_filter_small_terms)

        new_ia = Re_label_fermion_operators(self.qubitNo_re_label_dict, Sec_Quant_CC_ia_ops)
        new_ijab = Re_label_fermion_operators(self.qubitNo_re_label_dict, Sec_Quant_CC_ijab_ops)

        return new_ia, theta_parameters_ia, new_ijab, theta_parameters_ijab





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




from quchem.quantum_circuit_functions import *

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
                if list(ia_QubitOp_term.terms.keys())[0]:
                    Q_circuit_gen = full_exponentiated_PauliWord_circuit(ia_QubitOp_term, Theta_param_list_ia[ia_index])
                    Q_circuit = cirq.Circuit(cirq.decompose_once(
                        (Q_circuit_gen(*cirq.LineQubit.range(Q_circuit_gen.num_qubits())))))
                    Q_Circuit_generator_list.append(Q_circuit.all_operations())
        return Q_Circuit_generator_list

    def _Get_UCCSD_ijab_circuit(self, Theta_param_list_ijab):

        Q_Circuit_generator_list = []
        for ijab_index, iajb_QubitOp in enumerate(self.Qubit_Op_list_Second_Quant_CC_Ops_ijab):
            for ijab_QubitOp_term in iajb_QubitOp:
                if list(ijab_QubitOp_term.terms.keys())[0]:
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
#     Qubit_Op_list_Second_Quant_CC_Ops_ia, Qubit_Op_list_Second_Quant_CC_Ops_ijab = ansatz_obj.UCCSD_single_trotter_step(Sec_Quant_CC_ia_ops, Sec_Quant_CC_ijab_ops,
#                                                                                                                         transformation='JW')
#
#     full_ansatz_Q_Circ = Ansatz_Circuit(Qubit_Op_list_Second_Quant_CC_Ops_ia, Qubit_Op_list_Second_Quant_CC_Ops_ijab,
#                  Hamilt.molecule.n_qubits, Hamilt.molecule.n_electrons)
#
#     ansatz_cirq_circuit = full_ansatz_Q_Circ.Get_Full_HF_UCCSD_QC(theta_parameters_ia, theta_parameters_ijab, transformation='JW')
#
#     print(ansatz_cirq_circuit)