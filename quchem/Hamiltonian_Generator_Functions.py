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
        molecule = MolecularData(
            self.geometry,
            self.basis,
            self.multiplicity,
            description=self.MoleculeName)

        # Run Psi4.
        molecule = run_psi4(molecule,
                            run_scf=self.run_scf,
                            run_mp2=self.run_mp2,
                            run_cisd=self.run_cisd,
                            run_ccsd=self.run_ccsd,
                            run_fci=self.run_fci)

        self.molecule = molecule
        self.MolecularHamiltonian = molecule.get_molecular_hamiltonian()

        self.HF_Energy = molecule.hf_energy
        self.PSI4_FCI_Energy = molecule.fci_energy

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