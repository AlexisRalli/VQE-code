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



Constants_list = [X.QubitHamiltonian.terms[operations] for operations in X.QubitHamiltonian.terms]
qubitOpandNo_list = [operations for operations in X.QubitHamiltonian.terms]

num_qubits = X.MolecularHamiltonian.n_qubits
Q_list = [i for i in range(num_qubits)]

filled_list = []

for PauliWord, constant in X.QubitHamiltonian.terms.items():

    if len(PauliWord) == 0:
        constant_adder = constant
        #filled_list.append(constant_adder)     <-- may add... but need to change future funcitions
    else:
        qubits_indexed = [qubitNo for qubitNo, qubitOp in PauliWord]

        Not_indexed_qubits = [(qubit, 'I') for qubit in Q_list if qubit not in qubits_indexed]

        # Not in order (needs sorting)
        combined_terms_instance = [*PauliWord, *Not_indexed_qubits]

        filled_list.append(sorted(combined_terms_instance, key=lambda x: x[0]))

# checking!
print(filled_list)
print(constant_adder)

########
"""
From here can do TWO Things:

1. Get Qubit Hamiltonian MATRIX
   diagonalise it and obtain the HF energy!!!!
   
   
2. Get the QubitWise commiting GROUPS of the Hamiltonian

"""



##### TASK 1


# NOTE NEED TO CHECK THIS!!!!

"""
Note full_list variable from above contains qubitNo. and operations
e.g. [
        [(0, 'Z'), (1, 'I'), (2, 'I'), (3, 'I')],
        [(0, 'I'), (1, 'Z'), (2, 'I'), (3, 'I')],
        [(0, 'I'), (1, 'I'), (2, 'Z'), (3, 'I')]
    ]


NEXT need to take Kronecker product
"""

############

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

full_op_list=[]
for term in full_list:
    temp_list=[]
    for qubitNo, qubitOp in term:
        temp_list.append(OperatorsKeys[qubitOp])
    full_op_list.append(temp_list)


tensored = []
for oper in full_op_list:
    for i in range(len(oper)):

        if i == 0:
            TT =  np.kron(1, oper[i])
        else:
            TT =  np.kron(TT, oper[i])
    tensored.append(TT)


##### TASK 2
'''
Note Pauli Operators only commute with themselves and the identity and otherwise anti-commute!

'''

