import openfermion
import openfermioncirq
import cirq
import numpy as np

class Hamiltonian():

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
        self.FCI_Energy = None

        self.QubitHamiltonianCompleteTerms = None
        self.QubitOperator = np.zeros(1)
        self.full_CI_energy = None
        self.eig_values = None
        self.eig_vectors = None
        self.QWC_indices = None

    def Get_Molecular_Hamiltonian(self):
        from openfermion.hamiltonians import MolecularData
        from openfermionpsi4 import run_psi4
        delete_input = True
        delete_output = True


        if self.geometry == None:
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
        self.FCI_Energy = molecule.fci_energy

    def Get_Qubit_Hamiltonian_Openfermion(self):

        if self.MolecularHamiltonian == None:
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
        Function fills in identity terms in Qubit Hamiltonian


        (  [PauliWord, constant in self.QubitHamiltonian.terms.items()]   )

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

        :/param initial_state: A dictionary of Pauli words... keys are PauliWord and constant term is value
        :/type dict

        ...
        :raises [ErrorType]: [ErrorDescription]
        ...
        :return: A filled Operation list, with operation on each qubit
        :rtype: class
        """
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


    def Get_qubit_Hamiltonian_matrix(self):
        """
        e.g.

        Function turns Pauli letter list into SINGLE paulimatrix list... For example

        self.QubitHamiltonianCompleteTerms =
             [

                [(0, 'I'), (1, 'Z'), (2, 'I'), (3, 'I')],
                [(0, 'Y'), (1, 'X'), (2, 'X'), (3, 'Y')]
            ]

        becomes:

        PauliWord_list_matrices =

            [
                [ array([[1, 0],     array([[1, 0],    array([[1, 0],     array([[1, 0],
                         [0, 1]]),          [0, -1]]),         [0, 1]]),          [0, 1]]) ],

                [ array([[0, -1j],     array([[0, 1],    array([[0, 1],     array([[0, -1j],
                         [1j, 0]]),           [1, 0]]),         [1, 0]]),          [1j, 0]]) ]
            ]

        :/param initial_state: List Pauliwords (where each Pauliword is a list of tuples)
        :/type list

        ...
        :raises [ErrorType]: [ErrorDescription]
        ...
        :return: A list of Pauliwords, where each pauliword is a a list of numpy matrices
        :rtype: list
        """

        if self.QubitHamiltonianCompleteTerms == None:
            self.Get_Qubit_Hamiltonian_terms()


        OperatorsKeys = {
            'X': np.array([[0, 1],
                           [1, 0]]),
            'Y': np.array([[0, -1j],
                           [1j, 0]]),
            'Z': np.array([[1, 0],
                           [0, -1]]),
            'I': np.array([[1, 0],
                           [0, 1]]),
        }

        PauliWord_list_matrices = []
        for PauliWord in self.QubitHamiltonianCompleteTerms:
            PauliWord_matrix_instance = []
            for qubitNo, qubitOp in PauliWord:
                PauliWord_matrix_instance.append(OperatorsKeys[qubitOp])
            PauliWord_list_matrices.append(PauliWord_matrix_instance)

        # Next tensor together each row...:
        from functools import reduce

        tensored_terms = []
        for PauliWord_matrix in PauliWord_list_matrices:
            result1 = reduce((lambda single_QubitMatrix_FIRST, single_QubitMatrix_SECOND: np.kron(single_QubitMatrix_FIRST,
                                                                                                  single_QubitMatrix_SECOND)),
                                                                                                    PauliWord_matrix)
            tensored_terms.append(result1)

        # then multiply each matrix by constant
        Constants_list = [Constant for PauliWord, Constant in self.QubitHamiltonian.terms.items()]
        full_tensored_list = []
        for i in range(len(tensored_terms)):
            constant_factor = Constants_list[i]
            matrix_instance = tensored_terms[i]
            full_tensored_list.append(constant_factor * matrix_instance)


        QubitOperator = reduce((lambda first_matrix, second_matrix: first_matrix + second_matrix), full_tensored_list)

        self.QubitOperator = QubitOperator

    def Get_FCI_Energy(self):

        if np.array_equal(self.QubitOperator, np.zeros(1)):
            self.Get_qubit_Hamiltonian_matrix()

        eig_values, eig_vectors = np.linalg.eig(self.QubitOperator)

        FCI_Energy = min(eig_values)


        if FCI_Energy != self.FCI_Energy:
            # note self.FCI_Energy is PSI4 result!
            raise ValueError('Calculated FCI energy from Qubit Operator not equivalent to PSI4 calculation')

        self.full_CI_energy = FCI_Energy
        self.eig_values = eig_values
        self.eig_vectors = eig_vectors

    def Get_QWC_terms(self):
        """
        Function ... TODO


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
                (1, [0, [], 3]),
                (2, [0, [], []]),
                (3, [0, 1, []])
            ]

        """

        if self.QubitHamiltonianCompleteTerms == None:
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
                    QWC_indexes.append(j_list)

            commuting_Terms_indices = (i, QWC_indexes)

            index_of_commuting_terms.append(commuting_Terms_indices)

        self.QWC_indices = index_of_commuting_terms

    def Get_all_info(self):
        self.Get_Qubit_Hamiltonian_Openfermion()
        self.Get_qubit_Hamiltonian_matrix()
        self.Get_FCI_Energy()
        self.Get_QWC_terms()




if __name__ == '__main__':
    # X = Hamiltonian('H2')
    # X.Get_Qubit_Hamiltonian_Openfermion()
    # X.Get_qubit_Hamiltonian_matrix()
    # X.Get_FCI_Energy()
    # print(X.full_CI_energy)
    # X.Get_QWC_terms()
    # print(X.QWC_indices)

    Y = Hamiltonian('BeH2')
    Y.Get_Qubit_Hamiltonian_Openfermion()
    Y.Get_QWC_terms()
    print(Y.QWC_indices)




