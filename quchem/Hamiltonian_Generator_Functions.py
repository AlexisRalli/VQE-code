import numpy as np
from functools import reduce
from openfermion.hamiltonians import MolecularData
from openfermionpsi4 import run_psi4


def Get_commuting_indices(Pauliwords_string_list):
    """
    Method takes in qubit Hamiltonian as a list of Pauliwords that are lists of tuples (qubitNo, PauliString).
    Returns each index in qubit Hamiltonian and a list of corresponding indices that the PauliWord commutes with.

    Args:
        Pauliwords_string_list (list):

    Returns:
        Commuting_indices (list):

    Pauliwords_string_list =

        ['I0 I1 I2 I3',
         'Z0 I1 I2 I3',
         'I0 Z1 I2 I3',
         'I0 I1 Z2 I3',
         'I0 I1 I2 Z3',
         'Z0 Z1 I2 I3',
         'Y0 X1 X2 Y3',
         'Y0 Y1 X2 X3',
         'X0 X1 Y2 Y3',
         'X0 Y1 Y2 X3',
         'Z0 I1 Z2 I3',
         'Z0 I1 I2 Z3',
         'I0 Z1 Z2 I3',
         'I0 Z1 I2 Z3',
         'I0 I1 Z2 Z3'
         ]



    Returns a List of Tuples that have index of PauliWord and index of terms in the Hamiltonian that it commutes with

    index_of_commuting_terms =

        [(0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
         (1, [0, 2, 3, 4, 5, 10, 11, 12, 13, 14]),
         (2, [0, 1, 3, 4, 5, 10, 11, 12, 13, 14]),
         (3, [0, 1, 2, 4, 5, 10, 11, 12, 13, 14]),
         (4, [0, 1, 2, 3, 5, 10, 11, 12, 13, 14]),
         (5, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
         (6, [0, 5, 7, 8, 9, 10, 11, 12, 13, 14]),]

    """

    index_of_commuting_terms = []
    for i in range(len(QubitHamiltonianCompleteTerms)):
        Selected_PauliWord = QubitHamiltonianCompleteTerms[i]

        Complete_index_list = [index for index in range(len(QubitHamiltonianCompleteTerms)) if
                               index != i]  # all indexes except selected Pauli Word

        Commuting_indexes = []
        for j in Complete_index_list:
            j_list = []
            Comparison_PauliWord = QubitHamiltonianCompleteTerms[j]

            checker = [0 for i in range(len(Selected_PauliWord))]
            for k in range(len(Selected_PauliWord)):
                # compare tuples
                if Selected_PauliWord[k] == Comparison_PauliWord[k]:
                    checker[k] = 1

                # compare if identity present in selected P word OR of I present in comparison Pauli
                elif Selected_PauliWord[k][1] == 'I' or Comparison_PauliWord[k][1] == 'I':
                    checker[k] = 1

                else:
                    checker[k] = -1

            if reduce((lambda x, y: x * y), checker) == 1:  # <----- changing this to -ve one gives anti-commuting
                j_list.append(j)

            # if sum(checker) == self.MolecularHamiltonian.n_qubits:
            #     j_list.append(j)

            if j_list != []:
                Commuting_indexes.append(*j_list)
            else:
                # Commuting_indexes.append(j_list)      # <--- commented out! uneeded memory taken
                continue
        commuting_Terms_indices = (i, Commuting_indexes)

        index_of_commuting_terms.append(commuting_Terms_indices)

    return index_of_commuting_terms






class Hamiltonian():
    """

    The UCC_Terms object calculates and retains all the unitary coupled cluster terms.

    Args:
        MoleculeName (str): Name of Molecule
        run_scf (int, optional): Bool to run
        run_mp2 (int, optional):
        run_cisd (int, optional):
        run_ccsd (int, optional):
        run_fci (int, optional):
        basis (int, optional):
        multiplicity (int, optional):
        geometry (int, optional):

    Attributes:
        molecule (openfermion.hamiltonians._molecular_data.MolecularData):
        MolecularHamiltonian (openfermion.ops._interaction_operator.InteractionOperator): Molecular Hamiltonian

        QubitHamiltonian (openfermion.ops._qubit_operator.QubitOperator):
        HF_Energy (numpy.ndarray):
        PSI4_FCI_Energy (numpy.ndarray):


        QubitHamiltonianCompleteTerms (list):
        QubitOperator (scipy.sparse.csr.csr_matrix):
        full_CI_energy (numpy.complex128):
        eig_values (numpy.ndarray):
        eig_vectors (numpy.ndarray):
        QWC_indices (list):
        Commuting_indices (list):
        HamiltonainCofactors (list):

    """
    def __init__(self, MoleculeName,
                 run_scf = 1, run_mp2 = 1, run_cisd = 0, run_ccsd = 0, run_fci = 1,
                 basis = 'sto-3g',
                 multiplicity = 1,
                 geometry = None):

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
        self.MolecularHamiltonian = None

        self.QubitHamiltonian = None
        self.HF_Energy = None
        self.PSI4_FCI_Energy = None

        self.QubitHamiltonianCompleteTerms = None
        self.QubitOperator = np.zeros(1)
        self.full_CI_energy = None
        self.eig_values = None
        self.eig_vectors = None
        self.QWC_indices = None
        self.Commuting_indices = None
        self.HamiltonainCofactors = None

    def Get_Molecular_Hamiltonian(self):

        # delete_input = True
        # delete_output = True


        if self.geometry is None:
            self.Get_Geometry()


        # input
        self.molecule = MolecularData(
            self.geometry,
            self.basis,
            self.multiplicity,
            description=self.MoleculeName)

        # Run Psi4.
        self.molecule_Psi4 = run_psi4(self.molecule,
                            run_scf=self.run_scf,
                            run_mp2=self.run_mp2,
                            run_cisd=self.run_cisd,
                            run_ccsd=self.run_ccsd,
                            run_fci=self.run_fci)



        self.MolecularHamiltonian = self.molecule_Psi4.get_molecular_hamiltonian()

        self.HF_Energy = self.molecule_Psi4.hf_energy
        self.PSI4_FCI_Energy = self.molecule_Psi4.fci_energy

    def Get_Qubit_Hamiltonian_Openfermion(self):

        if self.MolecularHamiltonian is None:
            self.Get_Molecular_Hamiltonian()

        # Get Fermionic Hamiltonian
        from openfermion.transforms import get_fermion_operator
        fermionic_hamiltonian = get_fermion_operator(self.MolecularHamiltonian)

        # get qubit Hamiltonian
        from openfermion.transforms import jordan_wigner
        self.QubitHamiltonian = jordan_wigner(fermionic_hamiltonian)



    def Get_Geometry(self):

        from openfermion.utils import geometry_from_pubchem
        geometry = geometry_from_pubchem(self.MoleculeName)

        self.geometry = geometry

    def Get_Qubit_Hamiltonian_terms(self):

        """

        Method fills in identity terms in Qubit Hamiltonian


        Returns:
             QubitHamiltonianCompleteTerms (list): list of tuples of (qubitNo, PauliString)
             HamiltonainCofactors (list): List of cofactors from Hamiltonian


        PauliWords =
          [
              (),
              ((0, 'Z'),),
              ((1, 'Z'),),
              ((2, 'Z'),),
              ((0, 'Y'), (1, 'X'), (2, 'X'), (3, 'Y')),
          ]

        becomes:

        Operator_list_on_all_qubits =
          [
              [(0, 'Z'), (1, 'I'), (2, 'I'), (3, 'I')],
              [(0, 'I'), (1, 'Z'), (2, 'I'), (3, 'I')],
              [(0, 'I'), (1, 'I'), (2, 'Z'), (3, 'I')],
              [(0, 'Y'), (1, 'X'), (2, 'X'), (3, 'Y')]
          ]
        """

        if self.QubitHamiltonian is None:
            self.Get_Qubit_Hamiltonian_Openfermion()

        num_qubits = self.MolecularHamiltonian.n_qubits
        Q_list = [i for i in range(num_qubits)]


        Operator_list_on_all_qubits = []
        for PauliWord, constant in self.QubitHamiltonian.terms.items():

            if len(PauliWord) == 0:
                identity_on_all = [(qubit, 'I') for qubit in Q_list]
                Operator_list_on_all_qubits.append(identity_on_all)

            else:
                qubits_indexed = [qubitNo for qubitNo, qubitOp in PauliWord]

                Not_indexed_qubits = [(qubit, 'I') for qubit in Q_list if qubit not in qubits_indexed]

                # Not in order (needs sorting)
                combined_terms_instance = [*PauliWord, *Not_indexed_qubits]

                Operator_list_on_all_qubits.append(sorted(combined_terms_instance, key=lambda x: x[0]))

        self.QubitHamiltonianCompleteTerms = Operator_list_on_all_qubits
        self.HamiltonainCofactors = [constant for PauliWord, constant in self.QubitHamiltonian.terms.items()]


    def Get_qubit_Hamiltonian_matrix(self):
        """

        Method gets matrix of qubit Hamiltonian. Output is a sparse matrix of (2^num_qubits x 2^num_qubits)


        Returns:
             QubitOperator (scipy.sparse.csr.csr_matrix): Matrix of qubit operator. Use todense() method to get
                                                          full matrix

        .. code-block:: python
           :emphasize-lines: 18

           from quchem.Hamiltonian_Generator_Functions import *
            QubitHamiltonianCompleteTerms =  [
                                                [(0, 'I'), (1, 'Z'), (2, 'I'), (3, 'I')],
                                                [(0, 'Y'), (1, 'X'), (2, 'X'), (3, 'Y')]
                                            ]
            # PauliWord_list_matrices =
            #     [
            #         [ array([[1, 0],     array([[1, 0],    array([[1, 0],     array([[1, 0],
            #                  [0, 1]]),          [0, -1]]),         [0, 1]]),          [0, 1]]) ],
            #
            #         [ array([[0, -1j],     array([[0, 1],    array([[0, 1],     array([[0, -1j],
            #                  [1j, 0]]),           [1, 0]]),         [1, 0]]),          [1j, 0]]) ]
            #     ]


            # Each row is tensored together and mulitplied by cofactor. Then added together.

            >> <16x16 sparse matrix of type <class 'numpy.complex128'>
                with 32 stored elements in Compressed Sparse Row format>

            #using todense method:
            matrix([[ 0.13716573+0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                     -0.04919765+0.j],
                    [ 0.        +0.j,  0.13716573+0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.04919765+0.j,
                      0.        +0.j],
                    [ 0.        +0.j,  0.        +0.j,  0.13716573+0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j, -0.04919765+0.j,  0.        +0.j,
                      0.        +0.j],
                    [ 0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.13716573+0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.04919765+0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j],
                    [ 0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j, -0.13716573+0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j, -0.04919765+0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j],
                    [ 0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j, -0.13716573+0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.04919765+0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j],
                    [ 0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                     -0.13716573+0.j,  0.        +0.j,  0.        +0.j,
                     -0.04919765+0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j],
                    [ 0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j, -0.13716573+0.j,  0.04919765+0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j],
                    [ 0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.04919765+0.j,  0.13716573+0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j],
                    [ 0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                     -0.04919765+0.j,  0.        +0.j,  0.        +0.j,
                      0.13716573+0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j],
                    [ 0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.04919765+0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.13716573+0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j],
                    [ 0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j, -0.04919765+0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.13716573+0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j],
                    [ 0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.04919765+0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                     -0.13716573+0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j],
                    [ 0.        +0.j,  0.        +0.j, -0.04919765+0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j, -0.13716573+0.j,  0.        +0.j,
                      0.        +0.j],
                    [ 0.        +0.j,  0.04919765+0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j, -0.13716573+0.j,
                      0.        +0.j],
                    [-0.04919765+0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                      0.        +0.j,  0.        +0.j,  0.        +0.j,
                     -0.13716573+0.j]])
        """

        if self.QubitHamiltonianCompleteTerms is None:
            self.Get_Qubit_Hamiltonian_terms()


        from scipy.sparse import bsr_matrix

        X = bsr_matrix(np.array([[0, 1],
                                 [1, 0]])
                        )

        Y = bsr_matrix(np.array([[0, -1j],
                                 [1j, 0]])
                       )

        Z = bsr_matrix(np.array([[1, 0],
                                [0, -1]])
                       )
        I = bsr_matrix(np.array([[1, 0],
                                 [0, 1]])
                       )



        OperatorsKeys = {
            'X': X,
            'Y': Y,
            'Z': Z,
            'I': I,
        }



        PauliWord_list_matrices = []
        for PauliWord in self.QubitHamiltonianCompleteTerms:
            PauliWord_matrix_instance = []
            for qubitNo, qubitOp in PauliWord:
                PauliWord_matrix_instance.append(OperatorsKeys[qubitOp])
            PauliWord_list_matrices.append(PauliWord_matrix_instance)


    ############

        # Next tensor together each row...:
        from functools import reduce
        from tqdm import tqdm
        from scipy.sparse import csr_matrix
        QubitOperator = csr_matrix((2**self.MolecularHamiltonian.n_qubits,2**self.MolecularHamiltonian.n_qubits))

        Constants_list = [Constant for PauliWord, Constant in self.QubitHamiltonian.terms.items()]
        from scipy.sparse import kron
        for i in tqdm(range(len(PauliWord_list_matrices)), ascii=True, desc='Getting QubitOperator MATRIX'):
            PauliWord_matrix = PauliWord_list_matrices[i]


            # tensored_PauliWord = reduce((lambda single_QubitMatrix_FIRST, single_QubitMatrix_SECOND: kron(single_QubitMatrix_FIRST,
            #                                                                                       single_QubitMatrix_SECOND)),
            #                                                                                         PauliWord_matrix)
            tensored_PauliWord = reduce(kron, PauliWord_matrix)

            constant_factor = Constants_list[i]

            QubitOperator += constant_factor*tensored_PauliWord

        # notes to get operator in full form (from sparse) use to_dense() method!

        self.QubitOperator = QubitOperator


    def Get_FCI_Energy(self):
        """

        Method calculates energy from qubit operator (FCI).

        Raises:
            ValueError: FCI caculated from Qubit Operator does not match PSI4 calculation

        Returns:
            full_CI_energy (numpy.complex128):
            eig_values (numpy.ndarray):
            eig_vectors (numpy.ndarray):

        """
        if np.array_equal(self.QubitOperator, np.zeros(1)):
            self.Get_qubit_Hamiltonian_matrix()

        from scipy.sparse.linalg import eigs

        eig_values, eig_vectors = eigs(self.QubitOperator)

        FCI_Energy = min(eig_values)


        if not np.isclose(FCI_Energy.real, self.PSI4_FCI_Energy, rtol=1e-09, atol=0.0):
            # note self.FCI_Energy is PSI4 result!
            raise ValueError('Calculated FCI energy from Qubit Operator not equivalent to PSI4 calculation')

        # sorting to correct order
        idx = eig_values.argsort()[::-1]
        eig_values = eig_values[idx]
        eig_vectors = eig_vectors[:, idx]
        # HF state = self.eig_vectors[:,-1]

        self.full_CI_energy = FCI_Energy
        self.eig_values = eig_values
        self.eig_vectors = eig_vectors

    def Get_QWC_terms(self):
        """

        Method takes in qubit Hamiltonian as a list of Pauliwords that are lists of tuples (qubitNo, PauliString).
        Returns each index in qubit Hamiltonian and a list of corresponding indices that the PauliWord qubit wise
        commutes (QWC) with.

        Returns:
            QWC_indices (list):


        self.QubitHamiltonianCompleteTerms =

            [
                 [(0, 'I'), (1, 'I'), (2, 'I'), (3, 'I')],
                 [(0, 'Z'), (1, 'I'), (2, 'I'), (3, 'I')],
                 [(0, 'X'), (1, 'Y'), (2, 'Y'), (3, 'X')],
                 [(0, 'I'), (1, 'I'), (2, 'Z'), (3, 'Z')]
             ]

        Returns a List of Tuples that have index of PauliWord and index of terms in the Hamiltonian that it commutes with

        index_of_commuting_terms =

            [
                (0, [1, 2, 3]),
                (1, [0, 3]),
                (2, [0,]),
                (3, [0, 1])
            ]

        """

        if self.QubitHamiltonianCompleteTerms is None:
            self.Get_Qubit_Hamiltonian_terms()



        index_of_commuting_terms = []
        for i in range(len(self.QubitHamiltonianCompleteTerms)):
            Selected_PauliWord = self.QubitHamiltonianCompleteTerms[i]

            Complete_index_list = [index for index in range(len(self.QubitHamiltonianCompleteTerms)) if
                                   index != i]  # all indexes except selected Pauli Word

            QWC_indexes = []
            for j in Complete_index_list:
                j_list = []
                Comparison_PauliWord = self.QubitHamiltonianCompleteTerms[j]

                checker = [0 for i in range(len(Selected_PauliWord))]
                for k in range(len(Selected_PauliWord)):

                    # compare tuples
                    if Selected_PauliWord[k] == Comparison_PauliWord[k]:
                        checker[k] = 1

                    # compare if identity present in selected P word OR of I present in comparison Pauli
                    elif Selected_PauliWord[k][1] == 'I' or Comparison_PauliWord[k][1] == 'I':
                        checker[k] = 1

                if sum(checker) == self.MolecularHamiltonian.n_qubits:
                    j_list.append(j)

                if j_list != []:
                    QWC_indexes.append(*j_list)
                else:
                    # QWC_indexes.append(j_list)      # <--- commented out! uneeded memory taken
                    continue

            commuting_Terms_indices = (i, QWC_indexes)

            index_of_commuting_terms.append(commuting_Terms_indices)

        self.QWC_indices = index_of_commuting_terms


    def Get_commuting_indices(self):
        """
        Method takes in qubit Hamiltonian as a list of Pauliwords that are lists of tuples (qubitNo, PauliString).
        Returns each index in qubit Hamiltonian and a list of corresponding indices that the PauliWord commutes with.

        Returns:
            Commuting_indices (list):

        self.QubitHamiltonianCompleteTerms =

            [
                 [(0, 'I'), (1, 'I'), (2, 'I'), (3, 'I')],
                 [(0, 'Z'), (1, 'I'), (2, 'I'), (3, 'I')],
                 [(0, 'I'), (1, 'Z'), (2, 'I'), (3, 'I')],
                 [(0, 'I'), (1, 'I'), (2, 'Z'), (3, 'I')],
                 [(0, 'I'), (1, 'I'), (2, 'I'), (3, 'Z')],
                 [(0, 'Z'), (1, 'Z'), (2, 'I'), (3, 'I')],
                 [(0, 'Y'), (1, 'X'), (2, 'X'), (3, 'Y')]
             ]



        Returns a List of Tuples that have index of PauliWord and index of terms in the Hamiltonian that it commutes with

        index_of_commuting_terms =

            [(0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
             (1, [0, 2, 3, 4, 5, 10, 11, 12, 13, 14]),
             (2, [0, 1, 3, 4, 5, 10, 11, 12, 13, 14]),
             (3, [0, 1, 2, 4, 5, 10, 11, 12, 13, 14]),
             (4, [0, 1, 2, 3, 5, 10, 11, 12, 13, 14]),
             (5, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
             (6, [0, 5, 7, 8, 9, 10, 11, 12, 13, 14]),]

        """

        if self.QubitHamiltonianCompleteTerms is None:
            self.Get_Qubit_Hamiltonian_terms()



        index_of_commuting_terms = []
        for i in range(len(self.QubitHamiltonianCompleteTerms)):
            Selected_PauliWord = self.QubitHamiltonianCompleteTerms[i]

            Complete_index_list = [index for index in range(len(self.QubitHamiltonianCompleteTerms)) if
                                   index != i]  # all indexes except selected Pauli Word

            Commuting_indexes = []
            for j in Complete_index_list:
                j_list = []
                Comparison_PauliWord = self.QubitHamiltonianCompleteTerms[j]

                checker = [0 for i in range(len(Selected_PauliWord))]
                for k in range(len(Selected_PauliWord)):

                    # compare tuples
                    if Selected_PauliWord[k] == Comparison_PauliWord[k]:
                        checker[k] = 1

                    # compare if identity present in selected P word OR of I present in comparison Pauli
                    elif Selected_PauliWord[k][1] == 'I' or Comparison_PauliWord[k][1] == 'I':
                        checker[k] = 1

                    else:
                        checker[k] = -1

                if reduce((lambda x, y: x * y), checker) == 1:  # <----- changing this to -ve one gives anti-commuting
                    j_list.append(j)

                # if sum(checker) == self.MolecularHamiltonian.n_qubits:
                #     j_list.append(j)

                if j_list != []:
                    Commuting_indexes.append(*j_list)
                else:
                    #Commuting_indexes.append(j_list)      # <--- commented out! uneeded memory taken
                    continue
            commuting_Terms_indices = (i, Commuting_indexes)

            index_of_commuting_terms.append(commuting_Terms_indices)

        self.Commuting_indices = index_of_commuting_terms

    def Get_all_info(self, get_FCI_energy = False):

        """
        The UCC_Terms object calculates and retains all the unitary coupled cluster terms.

        Args:
            get_FCI_energy (bool, optional): Bool to calculate FCI energy, by diagonalising qubit hamiltonian matrix.
                                             note this is memory intensive!


        Returns:
            QWC_indices (list):
            Commuting_indices (list):

            QubitOperator (scipy.sparse.csr.csr_matrix, optional): Matrix of qubit operator. Use todense() method to get
                                                                      full matrix

            full_CI_energy(numpy.complex128, optional):
            eig_values(numpy.ndarray, optional):
            eig_vectors(numpy.ndarray, optional):

        """

        self.Get_Qubit_Hamiltonian_Openfermion()

        if get_FCI_energy == True:
            self.Get_qubit_Hamiltonian_matrix()
            self.Get_FCI_Energy()

        self.Get_QWC_terms()
        self.Get_commuting_indices()

if __name__ == '__main__':
    # X = Hamiltonian('H2')
    # X.Get_Qubit_Hamiltonian_Openfermion()
    # X.Get_qubit_Hamiltonian_matrix()
    # X.Get_FCI_Energy()
    # print(X.full_CI_energy)
    # X.Get_QWC_terms()
    # print(X.QWC_indices)

    Y = Hamiltonian('H2')
    Y.Get_all_info(get_FCI_energy=True)


    X = Hamiltonian('H2O')
    X.Get_all_info(get_FCI_energy=False)

    QWC_indices = X.QWC_indices
    Commuting_indices = X.Commuting_indices
    PauliWords = X.QubitHamiltonianCompleteTerms
    constants = X.HamiltonainCofactors


# def Get_qubit_Hamiltonian_matrix(QubitHamiltonianCompleteTerms, n_qubits, QubitHamiltonian):
#     from scipy.sparse import bsr_matrix
#
#     X = bsr_matrix(np.array([[0, 1],
#                              [1, 0]])
#                    )
#
#     Y = bsr_matrix(np.array([[0, -1j],
#                              [1j, 0]])
#                    )
#
#     Z = bsr_matrix(np.array([[1, 0],
#                              [0, -1]])
#                    )
#     I = bsr_matrix(np.array([[1, 0],
#                              [0, 1]])
#                    )
#
#     OperatorsKeys = {
#         'X': X,
#         'Y': Y,
#         'Z': Z,
#         'I': I,
#     }
#
#     PauliWord_list_matrices = []
#     for PauliWord in QubitHamiltonianCompleteTerms:
#         PauliWord_matrix_instance = []
#         for qubitNo, qubitOp in PauliWord:
#             PauliWord_matrix_instance.append(OperatorsKeys[qubitOp])
#         PauliWord_list_matrices.append(PauliWord_matrix_instance)
#
#     ############
#
#     # Next tensor together each row...:
#     from functools import reduce
#     from tqdm import tqdm
#     from scipy.sparse import csr_matrix
#     QubitOperator = csr_matrix((2 ** n_qubits, 2 ** n_qubits))
#
#     Constants_list = [Constant for PauliWord, Constant in QubitHamiltonian.terms.items()]
#     from scipy.sparse import kron
#     for i in tqdm(range(len(PauliWord_list_matrices)), ascii=True, desc='Getting QubitOperator MATRIX'):
#         PauliWord_matrix = PauliWord_list_matrices[i]
#
#         tensored_PauliWord = reduce(kron, PauliWord_matrix)
#
#         constant_factor = Constants_list[i]
#
#         QubitOperator += constant_factor * tensored_PauliWord
#
#     return QubitOperator
#
#
# QubitHamiltonianCompleteTerms = [
#     [(0, 'I'), (1, 'Z'), (2, 'I'), (3, 'I')],
#     [(0, 'Y'), (1, 'X'), (2, 'X'), (3, 'Y')]
# ]
# n_qubits = 4
# from openfermion.ops._qubit_operator import QubitOperator
#
# QubitHamiltonian = QubitOperator('Z1', 0.1371657293179602 + 0j) + QubitOperator('Y0 X1 X2 Y3',
#                                                                                 (0.04919764587885283 + 0j))
# answer = Get_qubit_Hamiltonian_matrix(QubitHamiltonianCompleteTerms, n_qubits, QubitHamiltonian)
# answer.todense()


class Hamiltonian():
    """

    The UCC_Terms object calculates and retains all the unitary coupled cluster terms.

    Args:
        MoleculeName (str): Name of Molecule
        run_scf (int, optional): Bool to run
        run_mp2 (int, optional):
        run_cisd (int, optional):
        run_ccsd (int, optional):
        run_fci (int, optional):
        basis (int, optional):
        multiplicity (int, optional):
        geometry (int, optional):

    Attributes:
        #TODO

    """
    def __init__(self, MoleculeName,
                 run_scf = 1, run_mp2 = 1, run_cisd = 1, run_ccsd = 1, run_fci = 1,
                 basis = 'sto-3g',
                 multiplicity = 1,
                 geometry = None):

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
        self.molecule_Psi4 = run_psi4(self.molecule,
                            run_scf=self.run_scf,
                            run_mp2=self.run_mp2,
                            run_cisd=self.run_cisd,
                            run_ccsd=self.run_ccsd,
                            run_fci=self.run_fci,
                            delete_input=False,
                            delete_output=False)

    def Get_Geometry(self):

        from openfermion.utils import geometry_from_pubchem
        geometry = geometry_from_pubchem(self.MoleculeName)

        self.geometry = geometry


    def Get_CCSD_Amplitudes(self):
        from openfermionpsi4._psi4_conversion_functions import parse_psi4_ccsd_amplitudes
        # https://github.com/quantumlib/OpenFermion-Psi4/blob/master/openfermionpsi4/_psi4_conversion_functions.py
        self.molecule.single_cc_amplitudes, self.molecule.double_cc_amplitudes = (
                                                                    parse_psi4_ccsd_amplitudes(
                                                                        2 * self.molecule.n_orbitals,
                                                                        self.molecule.get_n_alpha_electrons(),
                                                                        self.molecule.get_n_beta_electrons(),
                                                                        self.molecule.filename + ".out"))
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

    def Get_Molecular_Hamiltonian(self):
        if self.molecule is None:
            self.Run_Psi4()

        # H = constant + ∑_pq (h_pq a†_p a_q) + ∑_pqrs (h_pqrs a†_p a†_q a_r a_s)
        self.MolecularHamiltonian = self.molecule.get_molecular_hamiltonian() # instance of the MolecularOperator class
        self.singles_hamiltonian = self.MolecularHamiltonian.one_body_tensor # h_pq (n_qubits x n_qubits numpy array)
        self.doubles_hamiltonian = self.MolecularHamiltonian.two_body_tensor # h_pqrs (n_qubits x n_qubits x n_qubits x n_qubits numpy array


        # Get Matrix Form of QubitHamiltonian
        from openfermion.transforms import get_sparse_operator
        self.MolecularHamiltonianMatrix = get_sparse_operator(self.MolecularHamiltonian)


        # Get Fermionic Hamiltonian
        from openfermion.transforms import get_fermion_operator
        fermionic_hamiltonian = get_fermion_operator(self.MolecularHamiltonian)

        # get Qubit Hamiltonian
        from openfermion.transforms import jordan_wigner
        self.QubitHamiltonian = jordan_wigner(fermionic_hamiltonian)



    def  Get_Basis_state_in_occ_num_basis(self, occupied_orbitals_index_list=None):
        #Function to produce a basis state in the occupation number basis.

        #  aka input normally occupied_orbitals_index_list of HF state... eg |HF> = |0011>  --> INPUT = [0,1]
        # output is |HF> and <HF| state vectors under JW transform!

        import scipy
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

    def Get_ia_and_ijab_terms(self, Coupled_cluser_param=False, filter_small_terms = False): #TODO could add MP2 param option to initialise theta with MP2 amplitudes (rather than coupled cluster only option)

        # Sec_Quant_CC_ops is a list of fermionic creation and annihilation operators that perform UCCSD

        # e.g. for H2:
        #  Sec_Quant_CC_ops=  [
        #                      -1.0[0 ^ 2] + 1.0[2 ^ 0],            # -(a†0 a2) + (a†2 a0)
        #                      -1.0[1 ^ 3] + 1.0[3 ^ 1],            # -(a†1 a3) + (a†3 a1)
        #                      -1.0[0 ^ 1 ^ 2 3] + 1.0 [3^ 2^ 1 0]  # -(a†0 a†1 a2 a3) + a†3 a†2 a1 a0)
        #                     ]

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
                                    self.theta_parameters.append(self.molecule.double_cc_amplitudes[b][a][j][i])
                                else:
                                    self.theta_parameters.append(0)
                            self.Sec_Quant_CC_ops.append(two_elec)
                        else:
                            # NO filtering
                            two_elec = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0))) - \
                                       FermionOperator(((i, 1), (j, 1), (a, 0), (b, 0)))

                            if Coupled_cluser_param is True:
                                self.theta_parameters.append(self.molecule.double_cc_amplitudes[b][a][j][i])
                            else:
                                self.theta_parameters.append(0)

                            self.Sec_Quant_CC_ops.append(two_elec)

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

        return Sec_Quant_CC_ops, theta_parameters

class Hamiltonian_Transforms():
    def __init__(self, MolecularHamiltonian, Sec_Quant_CC_ops):
        self.MolecularHamiltonian = MolecularHamiltonian
        self.Sec_Quant_CC_ops = Sec_Quant_CC_ops

    def Get_Fermionic_Hamiltonian(self):
        #  Gives second quantised Hamiltonian
        # H = h00 a†0a0 + h11a†1a1 + h22a†2a2 +h33a†3a3 +
        #     h0110 a†0a†1a1a0 +h2332a†2a†3a3a2 + ... etc etc

        # note can get integrals from Get_CCSD_Amplitudes method of Hamiltonian class!

        from openfermion.transforms import get_fermion_operator
        FermionicHamiltonian = get_fermion_operator(self.MolecularHamiltonian)
        return FermionicHamiltonian

    def Get_Qubit_Hamiltonian_JW(self):
        #  Gives second quantised Hamiltonian under the JW transform!
        # H = h0 I + h1 Z0 + h2 Z1 +h3 Z2 + h4 Z3 + h5 Z0Z1 ... etc etc

        from openfermion.transforms import jordan_wigner
        FermionicHamiltonian = self.Get_Fermionic_Hamiltonian()
        QubitHamiltonian = jordan_wigner(FermionicHamiltonian)
        return QubitHamiltonian

    def Get_Jordan_Wigner_CC_Matrices(self):
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
            JW_CC_ops.append(get_sparse_operator(classical_op, n_qubits=4))
        return JW_CC_ops



if __name__ == '__main__':
    from quchem.Ansatz_Generator_Functions import *
    import numpy as np
    from functools import reduce
    from openfermion.hamiltonians import MolecularData
    from openfermionpsi4 import run_psi4

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

    Hamilt.Get_Molecular_Hamiltonian()
    SQ_CC_ops, THETA_params = Hamilt.Get_ia_and_ijab_terms(Coupled_cluser_param=True)
    print(SQ_CC_ops)

    HF_transformations = Hamiltonian_Transforms(Hamilt.MolecularHamiltonian, SQ_CC_ops)

    QubitHam = HF_transformations.Get_Qubit_Hamiltonian_JW()
    print(QubitHam)

    UCC_JW_excitation_matrix_list = HF_transformations.Get_Jordan_Wigner_CC_Matrices()

    HF_ref_ket, HF_ref_bra = Hamilt.Get_Basis_state_in_occ_num_basis(occupied_orbitals_index_list=[0,1]) # (|HF> , <HF|)

    # Hatree Fock Energy
    # H|HF> = E_hatree |HF>
    H_HatreeState = Hamilt.MolecularHamiltonianMatrix.dot(HF_ref_ket).toarray() #E*|HF> (all in one vecotr)
    HF_energy = np.dot(HF_ref_bra.toarray(), H_HatreeState)  #selects correct entries as in vector (look at H_HatreeState)  Energy!
    print('HF Energy from lin alg: ', HF_energy)

    # UCCSD Energy
    # UCCSD|HF> = E_UCCSD |HF>

    # NOTE going to use single trotter step!
    # also answer depends on THETA values!!

class CalcEnergy():
    def __init__(self, MolecularHamiltonianMatrix, reference_ket, n_qubits, JW_CC_ops_list):
        self.MolecularHamiltonianMatrix = MolecularHamiltonianMatrix
        self.reference_ket = reference_ket

        self.n_qubits = n_qubits
        self.JW_CC_ops_list = JW_CC_ops_list


    def Calc_HF_Energy(self):
        HF_ket = self.MolecularHamiltonianMatrix.dot(self.reference_ket).toarray()  # H |HF_ref> =   E*|HF> (all in one vecotr)
        HF_energy = np.dot(HF_ref_bra.toarray(), HF_ket)  # selects correct entries as in vector giving E (aka uses E |state vec>)
        print('HF Energy from lin alg: ', HF_energy)
        return HF_energy

    def Calc_UCCSD_No_Trot(self, parameters):
        # apply UCCSD matrix WITHOUT trotterisation!

        generator = scipy.sparse.csc_matrix((2 ** (self.n_qubits), 2 ** (self.n_qubits)), dtype=complex)
        for mat_op in range(0, len(self.JW_CC_ops_list)):
            generator = generator + parameters[mat_op] * self.JW_CC_ops_list[mat_op]
        new_state = scipy.sparse.linalg.expm_multiply(generator, self.reference_ket)
        new_bra = new_state.transpose().conj()
        assert (new_bra.dot(new_state).toarray()[0][0] - 1 < 0.0000001)
        energy = new_bra.dot(self.MolecularHamiltonianMatrix.dot(new_state))
        print('UCCSD WITHOUT trotterisation E: ', energy.toarray()[0][0].real)
        return energy.toarray()[0][0].real

    def Calc_UCCSD_with_Trot(self, parameters):
        # apply UCCSD matrix WITH first order trotterisation!

        new_state = self.reference_ket
        for k in reversed(range(0, len(parameters))):
            new_state = scipy.sparse.linalg.expm_multiply((parameters[k] * self.JW_CC_ops_list[k]), new_state)
        new_bra = new_state.transpose().conj()
        assert (new_bra.dot(new_state).toarray()[0][0] - 1 < 0.0000001)
        energy = new_bra.dot(self.MolecularHamiltonianMatrix.dot(new_state))
        print('UCCSD with trotterisation E: ', energy.toarray()[0][0].real)
        return energy.toarray()[0][0].real

w = CalcEnergy(Hamilt.MolecularHamiltonianMatrix, HF_ref_ket, Hamilt.molecule.n_qubits, UCC_JW_excitation_matrix_list)
w.Calc_HF_Energy()

THETA_params = [2.8, 2.1, 1]
w.Calc_UCCSD_No_Trot(THETA_params)
w.Calc_UCCSD_with_Trot(THETA_params)
w.Calc_UCCSD_with_Trot(THETA_params)


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



