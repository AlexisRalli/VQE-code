import openfermion
import openfermioncirq
import cirq


# MoleculeName = 'H2'
#
# def Get_Geometry(MoleculeName):
#
#     from openfermion.utils import geometry_from_pubchem
#     geometry = geometry_from_pubchem(MoleculeName)
#
#    return  geometry


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
            from openfermion.utils import geometry_from_pubchem
            self.geometry = geometry_from_pubchem(self.MoleculeName)



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


if __name__ == '__main__':
    X = Hamiltonian('H2')
    # X.Get_Qubit_Hamiltonian()
    # print(X.QubitHamiltonian)
    X.Get_Qubit_Hamiltonian_Terms()
    print(X.QubitHamiltonianTerms)

### Note maybe add:
# No_qubits = molecule.n_qubits
# line = list(range(0, No_qubits))
# line = cirq.LineQubit.range(No_qubits)




