import openfermion
import openfermioncirq
import cirq


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

        self.QubitHamiltonianTerms = []

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
        self.MolecularHamiltonian =  molecule.get_molecular_hamiltonian()

        self.HF_Energy = molecule.hf_energy
        self.FCI_Energy = molecule.fci_energy

    def Get_Qubit_Hamiltonian(self):

        if self.MolecularHamiltonian == None:
            self.Get_Molecular_Hamiltonian()

        # Get Fermionic Hamiltonian
        from openfermion.transforms import get_fermion_operator
        fermionic_hamiltonian = get_fermion_operator(self.MolecularHamiltonian)

        # get qubit Hamiltonian
        from openfermion.transforms import jordan_wigner
        self.QubitHamiltonian = jordan_wigner(fermionic_hamiltonian)


    def Get_Qubit_Hamiltonian_Terms(self):

        if self.QubitHamiltonian == None:
            self.Get_Qubit_Hamiltonian()

        for key, value in self.QubitHamiltonian.terms.items():
            self.QubitHamiltonianTerms.append((key, value))

    def Get_Geometry(self):

        from openfermion.utils import geometry_from_pubchem
        geometry = geometry_from_pubchem(self.MoleculeName)

        self.geometry = geometry


if __name__ == '__main__':
    X = Hamiltonian('H2')
    # X.Get_Qubit_Hamiltonian()
    # print(X.QubitHamiltonian)
    X.Get_Qubit_Hamiltonian_Terms()
    print(X.QubitHamiltonianTerms)


    print(X.QubitHamiltonianTerms)


### Note maybe add:
# No_qubits = molecule.n_qubits
# line = list(range(0, No_qubits))
# line = cirq.LineQubit.range(No_qubits)


###### Note:
#   X.QubitHamiltonian # This is QUBIT OPERATOR!
#   X.QubitHamiltonianTerms is a LIST!


def Get_Qubit_Hamiltonian_matrix(Hamiltonian_class):

    num_qubits = Hamiltonian_class.MolecularHamiltonian.n_qubits
    Q_list = [i for i in range(num_qubits)]

    ### this section adds Identity opertion on all qubits not operated on
    """
    (note Pauliword in loop is one instance of:)
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

    Operator_list_on_all_qubits = []
    for PauliWord, constant in Hamiltonian_class.QubitHamiltonian.terms.items():

        if len(PauliWord) == 0:

            identity_on_all = [(qubit, 'I') for qubit in Q_list]
            Operator_list_on_all_qubits.append(identity_on_all)

        else:
            qubits_indexed = [qubitNo for qubitNo, qubitOp in PauliWord]

            Not_indexed_qubits = [(qubit, 'I') for qubit in Q_list if qubit not in qubits_indexed]

            # Not in order (needs sorting)
            combined_terms_instance = [*PauliWord, *Not_indexed_qubits]

            Operator_list_on_all_qubits.append(sorted(combined_terms_instance, key=lambda x: x[0]))

    #print(Operator_list_on_all_qubits)



    # Next change make list of pauli matrices (not stings...)

    """
    e.g.
    Operator_list_on_all_qubits = 
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

    """

    import numpy as np

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
    for PauliWord in Operator_list_on_all_qubits:
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
    Constants_list = [Constant for PauliWord, Constant in Hamiltonian_class.QubitHamiltonian.terms.items()]
    full_tensored_list = []
    for i in range(len(tensored_terms)):
        constant_factor = Constants_list[i]
        matrix_instance = tensored_terms[i]
        full_tensored_list.append(constant_factor * matrix_instance)


    # Now find full Qubit Matrix
    QubitOperator = reduce((lambda first_matrix, second_matrix: first_matrix + second_matrix), full_tensored_list)


    eig_values, eig_vectors = np.linalg.eig(QubitOperator)

    FCI_Energy = min(eig_values)

    # print('FCI energy: ', FCI_Energy)
    # print(Hamiltonian_class.FCI_Energy)

    if FCI_Energy != Hamiltonian_class.FCI_Energy:
        raise ValueError('Calculated FCI energy from Qubit Operator not equivalent to PSI4 calculation')

    return FCI_Energy, QubitOperator





##### TASK 2
'''
Note Pauli Operators only commute with themselves and the identity and otherwise anti-commute!

'''


def QWC_Pauli_Operators(Hamiltonian_class):
    num_qubits = Hamiltonian_class.MolecularHamiltonian.n_qubits
    Q_list = [i for i in range(num_qubits)]

    ### this section adds Identity opertion on all qubits not operated on
    """
    (note Pauliword in loop is one instance of:)
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

    Operator_list_on_all_qubits = []
    for PauliWord, constant in Hamiltonian_class.QubitHamiltonian.terms.items():

        if len(PauliWord) == 0:

            identity_on_all = [(qubit, 'I') for qubit in Q_list]
            Operator_list_on_all_qubits.append(identity_on_all)

        else:
            qubits_indexed = [qubitNo for qubitNo, qubitOp in PauliWord]

            Not_indexed_qubits = [(qubit, 'I') for qubit in Q_list if qubit not in qubits_indexed]

            # Not in order (needs sorting)
            combined_terms_instance = [*PauliWord, *Not_indexed_qubits]

            Operator_list_on_all_qubits.append(sorted(combined_terms_instance, key=lambda x: x[0]))




    """

    Operator_list_on_all_qubits = 
        [
            [(0, 'Z'), (1, 'I'), (2, 'I'), (3, 'I')],
            [(0, 'I'), (1, 'Z'), (2, 'I'), (3, 'I')],
            [(0, 'I'), (1, 'I'), (2, 'Z'), (3, 'I')],
            [(0, 'Y'), (1, 'X'), (2, 'X'), (3, 'Y')]
        ]
        
    Returns a List of Tuples that have index and index of terms it commutes with
    
    [
         (0, [1, 2]),
         (1, [0, 2]),
         (2, [0, 1]),
         (3, [])  
    ]
    """

    index_of_commuting_terms=[]

    for i in range(len(Operator_list_on_all_qubits)):
        index_list_for_selected_P_word=[]
        Selected_PauliWord = Operator_list_on_all_qubits[i]

        Complete_index_list = [index for index in range(len(Operator_list_on_all_qubits)) if index != i] #all indexes except selected Pauli Word

        QWC_indexes =[]
        for j in Complete_index_list:
            j_list=[]
            Comparison_PauliWord = Operator_list_on_all_qubits[j]

            checker = [0 for i in range(len(Selected_PauliWord))]
            for k in range(len(Selected_PauliWord)):

                #compare tuples
                if Selected_PauliWord[k] == Comparison_PauliWord[k]:
                    #print('SAME Pauli STRING')
                    checker[k]=1
                    #print(Selected_PauliWord, 'the SAME as: ', Comparison_PauliWord)

                #compare if identity present AND also in comparison Pauli
                elif Selected_PauliWord[k][1] == 'I' or Comparison_PauliWord[k][1] == 'I':
                   checker[k]=1
                   #print(Selected_PauliWord, 'COMMUTES WITH: ', Comparison_PauliWord)

            if sum(checker) == num_qubits:
                j_list.append(j)

            if j_list != []:
                QWC_indexes.append(*j_list)
            else:
                QWC_indexes.append(j_list)


        commuting_Terms_indices = (i, QWC_indexes)

        index_of_commuting_terms.append(commuting_Terms_indices)

    return index_of_commuting_terms #, Operator_list_on_all_qubits



indices = QWC_Pauli_Operators(X)

