import numpy as np
import cirq
import random
import scipy



class UCCSD_Trotter_JW():
    """

    The UCCSD_Trotter object calculates trotterisation of UCCSD anatzse under JW transform

    Args:
        Second_Quant_CC_Ops_list (list): List of Fermionic Operators (openfermion.ops._fermion_operator.FermionOperator)
        Theta_param_list (list): List of Angles corresponding to UCCSD excitation amplitudes
    Attributes:
        #TODO

    """
    def __init__(self, Second_Quant_CC_Ops_list, Theta_param_list):
        self.Second_Quant_CC_Ops_list = Second_Quant_CC_Ops_list # FermionOperator
        self.Theta_param_list = Theta_param_list

    def SingleTrotterStep(self):

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

        Second_Quant_CC_JW_OP_list = []
        from openfermion.transforms import jordan_wigner
        for OP in self.Second_Quant_CC_Ops_list: # each OP = (T_i − T_i^†) i indicates if single, double, triple ... etc
            JW_OP = jordan_wigner(OP)
            Second_Quant_CC_JW_OP_list.append(JW_OP)
        return Second_Quant_CC_JW_OP_list

    def DoubleTrotterStep(self):
        #TODO
        # note for Theta_param_list, with 2nd order Trot will need to double angle params!
        pass

class UCCSD_Trotter_BK():
    """

    The UCCSD_Trotter object calculates trotterisation of UCCSD anatzse under BK transform

    Args:
        Second_Quant_CC_Ops_list (list): List of Fermionic Operators (openfermion.ops._fermion_operator.FermionOperator)
        Theta_param_list (list): List of Angles corresponding to UCCSD excitation amplitudes
    Attributes:
        #TODO

    """
    def __init__(self, Second_Quant_CC_Ops_list, Theta_param_list):
        self.Second_Quant_CC_Ops_list = Second_Quant_CC_Ops_list # FermionOperator
        self.Theta_param_list = Theta_param_list

    def SingleTrotterStep(self):

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

        Second_Quant_CC_JW_OP_list = []
        from openfermion.transforms import bravyi_kitaev
        for OP in self.Second_Quant_CC_Ops_list: # each OP = (T_i − T_i^†) i indicates if single, double, triple ... etc
            BK_OP = bravyi_kitaev(OP)
            Second_Quant_CC_JW_OP_list.append(BK_OP)
        return Second_Quant_CC_JW_OP_list

    def DoubleTrotterStep(self):
        #TODO
        # note for Theta_param_list, with 2nd order Trot will need to double angle params!
        pass

def Convert_QubitOperator_To_Pauliword_Str_list(Second_Quant_CC_OP_list):
    """
     From a list of QubitOperators (openfermion.ops._qubit_operator.QubitOperator) generate corresponding
     list of PauliStrings with cofactor!

    Args:
        Second_Quant_CC_OP_list (list): list of QubitOperators (openfermion.ops._qubit_operator.QubitOperator)

    Returns:
        PauliWord_str_Second_Quant_CC_OP_list (list): List of tuples (PauliWord_str, cofactor)

    e.g.
    Second_Quant_CC_OP_list =
    [
        -0.5j[X0 Z1 Y2] + 0.5j[Y0 Z1 X2],
        -0.5j[X1 Z2 Y3] +0.5j[Y1 Z2 X3],
        0.125j[X0 X1 X2 Y3] +0.125j[X0 X1 Y2 X3] + -0.125j[X0 Y1 X2 X3] + 0.125j[X0 Y1 Y2 Y3] +-0.125j[Y0 X1 X2 X3]
        +0.125j[Y0 X1 Y2 Y3] +-0.125j[Y0 Y1 X2 Y3] + -0.125j[Y0 Y1 Y2 X3]
    ]
    becomes

     [
        [('X0 Z1 Y2 I3', -0.5j), ('Y0 Z1 X2 I3', 0.5j)],
        [('I0 X1 Z2 Y3', -0.5j), ('I0 Y1 Z2 X3', 0.5j)],
        [('Y0 Y1 X2 Y3', -0.125j),('X0 X1 X2 Y3', 0.125j),('X0 Y1 Y2 Y3', 0.125j),('Y0 X1 Y2 Y3', 0.125j),
         ('X0 Y1 X2 X3', -0.125j),('Y0 X1 X2 X3', -0.125j),('Y0 Y1 Y2 X3', -0.125j),('X0 X1 Y2 X3', 0.125j)]
     ]
    """


    PauliWord_str_Second_Quant_CC_OP_list = []
    max_No_terms = max([len(list(QubitOP.terms.keys())[0]) for QubitOP in Second_Quant_CC_OP_list])
    all_indices = np.arange(0, max_No_terms, 1)

    for QubitOP in Second_Quant_CC_OP_list:
        T_Tdagg_Op_list = []

        for tupleOfTuples, factor in QubitOP.terms.items():
            qubit_OP_list = [tupl[1] + str(tupl[0]) for tupl in tupleOfTuples]

            if len(qubit_OP_list) < max_No_terms:
                # fill missing terms with Identity
                indices_present = [int(qubitNo_and_OP[1::]) for qubitNo_and_OP in qubit_OP_list]
                missing_indices = [index for index in all_indices if index not in indices_present]

                for index in missing_indices:
                    qubit_OP_list.append('I{}'.format(index))

                qubit_OP_list = sorted(qubit_OP_list, key=lambda x: int(x[1::])) # sort by qubitNo!
            # T_Tdagg_Op_list.append((qubit_OP_list, factor))

            seperator = ' '
            PauliWord = seperator.join(qubit_OP_list)
            T_Tdagg_Op_list.append((PauliWord, factor))
        PauliWord_str_Second_Quant_CC_OP_list.append(T_Tdagg_Op_list[::-1]) # reverse order (as need to do righthand side first!)
    return PauliWord_str_Second_Quant_CC_OP_list

class HF_state_generator():
    """
     Generate ground state HF state (singlet) in occupation number basis (canonical orbitals)

    Args:
        n_electrons (int): Number of electrons
        n_qubits (int): Number of qubits

    """

    def __init__(self, n_electrons, n_orbitals):
        self.n_electrons =n_electrons
        self.n_orbitals = n_orbitals

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

    def Get_JW_HF_state_in_occ_basis(self):
        # Note output order == np.array([*unoccupied,*occupied])
        from openfermion.utils._sparse_tools import jw_hartree_fock_state
        HF_state_vec = jw_hartree_fock_state(self.n_electrons, self.n_orbitals)
        HF_state_vec = HF_state_vec.reshape([2**self.n_orbitals,1])
        return self.Convert_basis_state_to_occ_num_basis(HF_state_vec)[::-1] # note change of order!

    def Beta_BK_matrix_transform(self):
        """
        Gives sparse Beta_BK_matrix matrix that transforms occupation number basis vectors of length n into the
        Bravyi-Kitaev basis.

        Based on  βn matrix in https://arxiv.org/pdf/1208.5986.pdf.


        Args:
            N_orbitals (int): Number of orbitals/qubits

        Returns:
            Beta_x (scipy.sparse.csr.csr_matrix): Sparse matrix

        """

        from scipy.sparse import csr_matrix, kron

        I = csr_matrix(np.eye(2))

        Beta_x = csr_matrix([1]).reshape([1, 1])

        for x in range(self.n_electrons):
            Beta_x = kron(I, Beta_x)

            Beta_x = Beta_x.tolil()
            Beta_x[0, :] = np.ones([1, Beta_x.shape[0]])
            Beta_x = Beta_x.tocsr()
        return Beta_x

    def Get_BK_HF_state_in_occ_basis(self):

        mat_transform = self.Beta_BK_matrix_transform()

        Hartree_Fock_JW_occ_basis_state = self.Get_JW_HF_state_in_occ_basis()
        Hartree_Fock_JW_occ_basis_state = np.array(Hartree_Fock_JW_occ_basis_state).reshape([len(Hartree_Fock_JW_occ_basis_state),1])

        HF_state_BK_basis = mat_transform.dot(Hartree_Fock_JW_occ_basis_state) % 2

        return HF_state_BK_basis

    def Get_BK_HF_vector(self):
        zero = np.array([[1], [0]])
        one = np.array([[0], [1]])

        BK_HF_occ_Basis = self.Get_BK_HF_state_in_occ_basis().reshape([len(self.Get_BK_HF_state_in_occ_basis())])

        from numpy import kron
        from functools import reduce
        STATE=[]
        for bit in BK_HF_occ_Basis[::-1]: # note reverse order!!
            if int(bit)==0:
                STATE.append(zero)
            elif int(bit)==1:
                STATE.append(one)
            else:
                raise ValueError('invalid state bit is not zero or one but: {}'.format(bit))
        STATE_vec = reduce(kron, STATE)
        return STATE_vec

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
    def __init__(self, PauliWord_str_Second_Quant_CC_JW_OP_list, n_electrons, n_qubits):
        self.PauliWord_str_Second_Quant_CC_JW_OP_list = PauliWord_str_Second_Quant_CC_JW_OP_list
        self.n_qubits = n_qubits
        self.n_electrons = n_electrons

        self.HF_QCirc = None

    def Get_HF_Quantum_Circuit(self):
        HF_state = HF_state_generator(self.n_electrons, self.n_qubits)
        HF_state_list = HF_state.Get_JW_HF_state_in_occ_basis()
        HF_state_prep = State_Prep(HF_state_list)
        HF_state_prep_circuit = cirq.Circuit(cirq.decompose_once(
            (HF_state_prep(*cirq.LineQubit.range(HF_state_prep.num_qubits())))))
        self.HF_QCirc = list(HF_state_prep_circuit.all_operations())

    def Get_UCCSD_Quantum_Circuit(self, Theta_param_list):

        Q_Circuit_generator_list =[]

        for i in range(len(self.PauliWord_str_Second_Quant_CC_JW_OP_list)):
            ExcitationOp = self.PauliWord_str_Second_Quant_CC_JW_OP_list[i]
            Theta = Theta_param_list[i]
            for Paulistring_and_Cofactor in ExcitationOp:
                Q_circuit_gen = full_exponentiated_PauliWord_circuit(Paulistring_and_Cofactor, Theta)
                Q_circuit = cirq.Circuit(cirq.decompose_once(
                    (Q_circuit_gen(*cirq.LineQubit.range(Q_circuit_gen.num_qubits())))))
                Q_Circuit_generator_list.append(Q_circuit.all_operations())
        return Q_Circuit_generator_list

    def Get_Full_HF_UCCSD_QC(self, Theta_param_list):

        if self.HF_QCirc is None:
            self.Get_HF_Quantum_Circuit()

        UCCSD_QC_List = self.Get_UCCSD_Quantum_Circuit(Theta_param_list)

        full_circuit = cirq.Circuit(
            [
                self.HF_QCirc,
                *UCCSD_QC_List,
            ]
        )
        return full_circuit

if __name__ == '__main__':
    ####### REQUIRED TO USE ANSATZSE CLASS ######
    from quchem.Hamiltonian_Generator_Functions import *
    ### Variable Parameters
    Molecule = 'H2'
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.74))]
    num_shots = 10000
    ####

    ### Get Hamiltonian
    Hamilt = Hamiltonian(Molecule,
                         run_scf=1, run_mp2=1, run_cisd=1, run_ccsd=1, run_fci=1,
                         basis='sto-3g',
                         multiplicity=1,
                         geometry=geometry)  # normally None!

    Hamilt.Get_Molecular_Hamiltonian(Get_H_matrix=True)
    SQ_CC_ops, THETA_params = Hamilt.Get_ia_and_ijab_terms(Coupled_cluser_param=False)
    print(SQ_CC_ops)

    HF_transformations = Hamiltonian_Transforms(Hamilt.MolecularHamiltonian, SQ_CC_ops, Hamilt.molecule.n_qubits)
    # fermionic_hamiltonian = HF_transformations.Get_Fermionic_Hamiltonian()
    # print(fermionic_hamiltonian)
    Qubit_Hamiltonian = HF_transformations.Get_Qubit_Hamiltonian_JW() # qubit Hamiltonian version of Molecular Hamiltonian
    print(Qubit_Hamiltonian)
    ####### FINISHED   #### REQUIRED TO USE ANSATZSE CLASS ######

    UCCSD = UCCSD_Trotter_JW(SQ_CC_ops, THETA_params)

    Second_Quant_CC_JW_OP_list = UCCSD.SingleTrotterStep()

    PauliWord_list = Convert_QubitOperator_To_Pauliword_Str_list(Second_Quant_CC_JW_OP_list)
    HF_UCCSD_ansatz = Ansatz_Circuit(PauliWord_list, Hamilt.molecule.n_electrons, Hamilt.molecule.n_qubits)

    THETA_params = [np.pi, 3*np.pi, 2*np.pi]
    # THETA_params = [random.uniform(0, 2 * np.pi) for _ in range(Hamilt.num_theta_parameters)]
    ansatz_Q_cicuit = HF_UCCSD_ansatz.Get_Full_HF_UCCSD_QC(THETA_params)
    #print(ansatz_Q_cicuit)

class Ansatz_MATRIX():
    """

    Build the ansatz state through linear algebra rather than quantum circuits.

    Args:
        PauliWord_str_Second_Quant_CC_JW_OP_list (list): List of Fermionic Operators (openfermion.ops._fermion_operator.FermionOperator)
        n_electrons (int): Number of electrons
        n_qubits (int): Number of qubits

    Attributes:
        reference_ket ():
        UCCSD_ops_matrix_list ():

    """
    def __init__(self, Second_Quant_CC_JW_OP_list, n_electrons, n_qubits):
        self.Second_Quant_CC_JW_OP_list = Second_Quant_CC_JW_OP_list
        self.n_qubits = n_qubits
        self.n_electrons = n_electrons

        self.reference_ket = None
        self.UCCSD_ops_matrix_list = None

    def _Get_Basis_state_in_occ_num_basis(self, occupied_orbitals_index_list):
        """

        Method to obtain basis state under JW transform of state defined in occupation number basis.
        e.g. for H2 under the Jordan Wigner transfrom has |HF> = |0011> in occ no. basis
        occupied_orbitals_index_list = [0,1] <- as first orbitals occupied

        These outputs (|HF> and <HF|) can be used with MolecularHamiltonianMatrix!.

        Args:
            occupied_orbitals_index_list (list): list of orbital indices that are OCCUPIED

        returns:
            reference_ket (scipy.sparse.csr.csr_matrix): Sparse matrix of KET corresponding to occ no basis state under
                                                         JW transform


        """

        from openfermion import jw_configuration_state

        reference_ket = scipy.sparse.csc_matrix(jw_configuration_state(occupied_orbitals_index_list,
                                                                       self.n_qubits)).transpose()
        # reference_bra = reference_ket.transpose().conj()
        self.reference_ket = reference_ket

    def _Get_UCCSD_matrices(self):
        from openfermion.transforms import get_sparse_operator
        UCCSD_ops_matrix_list = []
        for classical_op in self.Second_Quant_CC_JW_OP_list:
            # matrix operator of coupled cluster operations
            UCCSD_ops_matrix_list.append(get_sparse_operator(classical_op, n_qubits=self.n_qubits))
        self.UCCSD_ops_matrix_list = UCCSD_ops_matrix_list

    def Calc_ansatz_state_WITH_trot(self, parameters):

        if self.UCCSD_ops_matrix_list is None:
             self._Get_UCCSD_matrices()

        new_state = self.reference_ket
        for k in reversed(range(0, len(parameters))):
            new_state = scipy.sparse.linalg.expm_multiply((parameters[k] * self.UCCSD_ops_matrix_list[k]), new_state)
        # bra = new_state.transpose().conj()
        return new_state

    def Calc_ansatz_state_withOUT_trot(self, parameters):

        if self.UCCSD_ops_matrix_list is None:
            self._Get_UCCSD_matrices()

        generator = scipy.sparse.csc_matrix((2 ** (self.n_qubits), 2 ** (self.n_qubits)), dtype=complex)
        for mat_op in range(0, len(self.UCCSD_ops_matrix_list)):
            generator = generator + parameters[mat_op] * self.UCCSD_ops_matrix_list[mat_op]
        new_state = scipy.sparse.linalg.expm_multiply(generator, self.reference_ket)
        # new_bra = new_state.transpose().conj()
        return new_state

if __name__ == '__main__':
    XX = Ansatz_MATRIX(Second_Quant_CC_JW_OP_list, Hamilt.molecule.n_electrons, Hamilt.molecule.n_qubits,)
    XX._Get_Basis_state_in_occ_num_basis([0, 1])
    state_with_T = XX.Calc_ansatz_state_WITH_trot([1, 2, 3])
    state_without_T = XX.Calc_ansatz_state_withOUT_trot([1, 2, 3])

class Qubit_Hamiltonian_MATRIX(Ansatz_MATRIX):
    """

    Build the ansatz state through linear algebra rather than quantum circuits.

    Args:
        PauliWord_str_Second_Quant_CC_JW_OP_list (list): List of Fermionic Operators (openfermion.ops._fermion_operator.FermionOperator)
        n_electrons (int): Number of electrons
        n_qubits (int): Number of qubits
        QubitOperator ( openfermion.ops._qubit_operator.QubitOperator):

    Attributes:
        Qubit_Ham_matrix ():
        occupied_orbitals_index_list ():

    QubitOperator =
                (-0.09706626861762581+0j) [] +
                (-0.045302615508689394+0j) [X0 X1 Y2 Y3] +
                (0.045302615508689394+0j) [X0 Y1 Y2 X3] +
                (0.045302615508689394+0j) [Y0 X1 X2 Y3] +
                (-0.045302615508689394+0j) [Y0 Y1 X2 X3] +
                (0.17141282639402383+0j) [Z0] +
                (0.168688981686933+0j) [Z0 Z1] +
                (0.12062523481381841+0j) [Z0 Z2] +
                (0.16592785032250779+0j) [Z0 Z3] +
                (0.1714128263940239+0j) [Z1] +
                (0.16592785032250779+0j) [Z1 Z2] +
                (0.12062523481381841+0j) [Z1 Z3] +
                (-0.22343153674663985+0j) [Z2] +
                (0.1744128761065161+0j) [Z2 Z3] +
                (-0.22343153674663985+0j) [Z3]

    """
    def __init__(self, Second_Quant_CC_JW_OP_list, n_electrons, n_qubits, QubitOperator,
                 occupied_orbitals_index_list):
        super().__init__(Second_Quant_CC_JW_OP_list, n_electrons, n_qubits)
        self.QubitOperator=QubitOperator

        self.Qubit_Ham_matrix = None
        self.occupied_orbitals_index_list = occupied_orbitals_index_list

    def _get_qubit_hamiltonian_matrix(self):
        from openfermion.transforms import get_sparse_operator
        self.Qubit_Ham_matrix = get_sparse_operator(self.QubitOperator)

    def find_energy_WITH_trot(self,parameters):
        if self.Qubit_Ham_matrix is None:
            self._get_qubit_hamiltonian_matrix()

        self._Get_Basis_state_in_occ_num_basis(self.occupied_orbitals_index_list)

        ket = self.Calc_ansatz_state_WITH_trot(parameters)
        bra = ket.transpose().conj()

        energy = bra.dot(self.Qubit_Ham_matrix.dot(ket))
        return energy.toarray()[0][0].real

    def find_energy_withOUT_trot(self, parameters):
        if self.Qubit_Ham_matrix is None:
            self._get_qubit_hamiltonian_matrix()

        self._Get_Basis_state_in_occ_num_basis(self.occupied_orbitals_index_list)

        ket = self.Calc_ansatz_state_withOUT_trot(parameters)
        bra = ket.transpose().conj()

        energy = bra.dot(self.Qubit_Ham_matrix.dot(ket))
        return energy.toarray()[0][0].real

if __name__ == '__main__':
    orb_ind = [0, 1]
    YY = Qubit_Hamiltonian_MATRIX(Second_Quant_CC_JW_OP_list, Hamilt.molecule.n_electrons, Hamilt.molecule.n_qubits,
                                  Qubit_Hamiltonian, orb_ind)
    params = [7.21692414e-05, 5.35729301e-03, 3.25177337e+00] #[1,2,3]
    YY.find_energy_withOUT_trot(params)
    YY.find_energy_WITH_trot(params)

if __name__ == '__main__':
    # Test on LiH
    from quchem.Hamiltonian_Generator_Functions import *

    ### Variable Parameters
    Molecule = 'LiH'
    geometry = None
    ####

    ### Get Hamiltonian
    Hamilt = Hamiltonian(Molecule,
                         run_scf=1, run_mp2=1, run_cisd=1, run_ccsd=1, run_fci=1,
                         basis='sto-3g',
                         multiplicity=1,
                         geometry=geometry)  # normally None!

    Hamilt.Get_Molecular_Hamiltonian(Get_H_matrix=False)
    SQ_CC_ops, THETA_params = Hamilt.Get_ia_and_ijab_terms(Coupled_cluser_param=False)

    HF_transformations = Hamiltonian_Transforms(Hamilt.MolecularHamiltonian, SQ_CC_ops, Hamilt.molecule.n_qubits)
    Qubit_Hamiltonian = HF_transformations.Get_Qubit_Hamiltonian_JW()  # qubit

    UCCSD = UCCSD_Trotter_JW(SQ_CC_ops, THETA_params)

    Second_Quant_CC_JW_OP_list = UCCSD.SingleTrotterStep()

    orbital_index_list = [0, 1, 2]
    YY = Qubit_Hamiltonian_MATRIX(Second_Quant_CC_JW_OP_list, Hamilt.molecule.n_electrons, Hamilt.molecule.n_qubits,
                                  Qubit_Hamiltonian, orbital_index_list)
    YY.find_energy_withOUT_trot(THETA_params)
    YY.find_energy_WITH_trot(THETA_params)

    # from quchem.Scipy_Optimizer import *
    # GG = Optimizer(YY.find_energy_WITH_trot, THETA_params, 'Nelder-Mead', store_values=True, display_iter_steps=True,
    #                tol=1e-9,
    #                display_convergence_message=True)
    # GG.get_env(100)

def Beta_BK_matrix(N_orbitals):
    """
    Gives sparse Beta_BK_matrix matrix that transforms occupation number basis vectors of length n into the
    Bravyi-Kitaev basis.

    Based on  βn matrix in https://arxiv.org/pdf/1208.5986.pdf.


    Args:
        N_orbitals (int): Number of orbitals/qubits

    Returns:
        Beta_x (scipy.sparse.csr.csr_matrix): Sparse matrix

    """
    from scipy.sparse import csr_matrix, kron

    I = csr_matrix(np.eye(2))

    Beta_x = csr_matrix([1]).reshape([1, 1])

    for x in range(N_orbitals):
        Beta_x = kron(I,Beta_x)

        Beta_x = Beta_x.tolil()
        Beta_x[0, :] = np.ones([1, Beta_x.shape[0]])
        Beta_x = Beta_x.tocsr()
    return Beta_x

