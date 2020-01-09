import numpy as np
import scipy

from openfermion.hamiltonians import MolecularData
from openfermionpsi4 import run_psi4


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


if __name__ == '__main__':

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
    print('Qubit Hamiltonian: ', QubitHam)

    UCC_JW_excitation_matrix_list = HF_transformations.Get_Jordan_Wigner_CC_Matrices()

    HF_ref_ket, HF_ref_bra = Hamilt.Get_Basis_state_in_occ_num_basis(occupied_orbitals_index_list=[0,1]) # (|HF> , <HF|)

    # Hatree Fock Energy
    # H|HF> = E_hatree |HF>
    H_HatreeState = Hamilt.MolecularHamiltonianMatrix.dot(HF_ref_ket).toarray() #E*|HF> (all in one vecotr)
    HF_energy = np.dot(HF_ref_bra.toarray(), H_HatreeState)  #selects correct entries as in vector (look at H_HatreeState)  Energy!
    print('HF Energy from lin alg: ', HF_energy)


    w = CalcEnergy(Hamilt.MolecularHamiltonianMatrix, HF_ref_ket, Hamilt.molecule.n_qubits,
                   UCC_JW_excitation_matrix_list)
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



