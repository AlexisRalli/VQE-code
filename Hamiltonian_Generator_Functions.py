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
    return Operator_list_on_all_qubits




complete_list = QWC_Pauli_Operators(X)
num_qubits = 4
reverse_column_index = [column_i for column_i in range(len(complete_list))][::-1]


commuting_Pauli_words=[]
for column_i in reverse_column_index:

    commuting_Pauli_word_set=[]

    complete_list = QWC_Pauli_Operators(X)
    Selected_PauliWord = np.array(complete_list.pop(column_i))

    commuting_Pauli_word_set.append(Selected_PauliWord)

    #complete_list no longer complete!

    for ID in range(len(complete_list)):

        commuting_parts = []

        PauliWord = np.array(complete_list[ID])

        for qubit_index in range(num_qubits):
            # comutes with itself
            if np.array_equal(PauliWord[qubit_index], Selected_PauliWord[qubit_index]):
                #print('selected PauliWord: ', Selected_PauliWord[qubit_index], 'is equiv to: ', PauliWord[qubit_index])

                commuting_parts.append(PauliWord[qubit_index])

                #commuting_parts.append(PauliWord[qubit_index])
            elif np.array_equal(PauliWord[qubit_index], np.array(['{}'.format(qubit_index), 'I'])) and bool(np.array(['{}'.format(qubit_index), 'I']) in PauliWord[qubit_index]):
                #print('selected PauliWord: ', Selected_PauliWord[qubit_index], 'COMMUTES WITH: ',
                #      PauliWord[qubit_index])
                commuting_parts.append(PauliWord[qubit_index])

        if len(commuting_parts) == num_qubits:
            commuting_Pauli_word_set.append(commuting_parts)

            print(commuting_Pauli_word_set)
    commuting_Pauli_words.append(commuting_Pauli_word_set)


index_list=[]
complete_list = QWC_Pauli_Operators(X)
num_qubits = 4

for i in range(len(complete_list)):
    Selected_PauliWord = complete_list[i]

    indexes=[]
    for j in range(i + 1, len(complete_list)):
        Comparison_PauliWord = complete_list[j]

        QWC_qubits=[]
        for qubit_index in range(num_qubits):
            if Selected_PauliWord[qubit_index] == Comparison_PauliWord[qubit_index]:
                QWC_qubits.append(j)
                print(Selected_PauliWord[qubit_index], 'IS', Comparison_PauliWord[qubit_index])

            if Selected_PauliWord[qubit_index] == (qubit_index, 'I') and bool((qubit_index, 'I') in [string for P_word in complete_list for string in P_word]):
                QWC_qubits.append(j)
                print(Selected_PauliWord[qubit_index], 'COMMUTES', Comparison_PauliWord[qubit_index])

        index_list.append([i, j])




complete_list = ['cat', 'dog', 'goat', 'cat', 'bat', 'dog', 'wolf']
for i in range(len(complete_list)):
    for j in range(i + 1, len(complete_list)):
        if complete_list[i] == complete_list[j]:
            print(complete_list[i])

counter = 0
complete_list = ['cat', 'dog', 'goat', 'cat', 'bat', 'dog', 'wolf']
for i in range(len(complete_list)):
    for j in range(i + 1, len(complete_list)):
        print(complete_list[i], complete_list[j])
        counter+=1
print(counter)