from quchem.Hamiltonian_Generator_Functions import Hamiltonian
from quchem.Graph import BuildGraph_string
from quchem.Unitary_partitioning import *
from quchem.Ansatz_Generator_Functions import *


### Variable Parameters
Molecule = 'H2'
geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.74))]
n_electrons = 2
num_shots = 10000
####

### Get Hamiltonian
Hamilt = Hamiltonian(Molecule,
                     run_scf = 1, run_mp2 = 1, run_cisd = 0, run_ccsd = 0, run_fci = 1,
                 basis = 'sto-3g',
                 multiplicity = 1,
                 geometry = geometry) # normally None!

Hamilt.Get_all_info(get_FCI_energy=False)

###### Build Graph --> to get anti-commuting sets

Commuting_indices = Hamilt.Commuting_indices
PauliWords = Hamilt.QubitHamiltonianCompleteTerms
constants = Hamilt.HamiltonainCofactors

HamiltGraph = BuildGraph_string(PauliWords, Commuting_indices, constants)
HamiltGraph.Build_string_nodes()  # plot_graph=True)
HamiltGraph.Build_string_edges()  # plot_graph=True)
HamiltGraph.Get_complementary_graph_string()  # plot_graph=True)
HamiltGraph.colouring(plot_graph=False)

anti_commuting_sets = HamiltGraph.anticommuting_sets

### build Hamiltonian

xx = X_sk_terms(anti_commuting_sets, S=0)
xx.Get_all_X_sk_operators()
xx.X_sk_Ops

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

from functools import reduce
from scipy.sparse import csr_matrix
from scipy.sparse import kron


### next want to build matrix!

n_qubits = 4

for key in anti_commuting_sets:
    if key not in xx.X_sk_Ops:
        PauliWord_constant = anti_commuting_sets[key]

        # next go through and tensor together with PauliWords
    else:
        R_SL_DAGGER_matrix_LIST =[]
        R_SL_matrix_LIST = []
        for SL_terms in xx.X_sk_Ops[key]['X_sk_theta_sk']:
            theta_sk = SL_terms['theta_sk']

            R_S_DAGGER = My_R_sk_Gate(theta_sk, dagger=True)
            R_S_DAGGER_matrix = R_S_DAGGER._unitary_()
            R_SL_DAGGER_matrix_LIST.append(R_S_DAGGER_matrix)

            R_S = My_R_sk_Gate(theta_sk, dagger=False)
            R_S_matrix = R_S._unitary_()
            R_SL_matrix_LIST.append(R_S_matrix)


        Pauliword_S = xx.X_sk_Ops[key]['PauliWord_S']
        beta_s = Pauliword_S[1]

        Paulis_to_be_tensored=[]

        for PauliString in Pauliword_S[0].split(' '):
            Op = OperatorsKeys[PauliString[0]]
            Paulis_to_be_tensored.append(Op)

        # next perform tensor!
        Operator = csr_matrix((2 **n_qubits, 2 **n_qubits))

        tensored_PauliWord = reduce(kron, Paulis_to_be_tensored)
        tensored_R_SL_DAGGER_matrix_LIST = reduce(kron, R_SL_DAGGER_matrix_LIST)
        tensored_R_SL_matrix_LIST = reduce(kron, R_SL_matrix_LIST)

        Operator += beta_s * tensored_PauliWord



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

        QubitOperator = csr_matrix((2 ** self.MolecularHamiltonian.n_qubits, 2 ** self.MolecularHamiltonian.n_qubits))

        Constants_list = [Constant for PauliWord, Constant in self.QubitHamiltonian.terms.items()]
        from scipy.sparse import kron

        for i in tqdm(range(len(PauliWord_list_matrices)), ascii=True, desc='Getting QubitOperator MATRIX'):
            PauliWord_matrix = PauliWord_list_matrices[i]

            # tensored_PauliWord = reduce((lambda single_QubitMatrix_FIRST, single_QubitMatrix_SECOND: kron(single_QubitMatrix_FIRST,
            #                                                                                       single_QubitMatrix_SECOND)),
            #                                                                                         PauliWord_matrix)
            tensored_PauliWord = reduce(kron, PauliWord_matrix)

            constant_factor = Constants_list[i]

            QubitOperator += constant_factor * tensored_PauliWord

        # notes to get operator in full form (from sparse) use to_dense() method!

        self.QubitOperator = QubitOperator


