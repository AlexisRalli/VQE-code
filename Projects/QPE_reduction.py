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
from scipy.linalg import expm

from tqdm import tqdm

### next want to build matrix!

n_qubits = 4
Operator = csr_matrix((2 **n_qubits, 2 **n_qubits))

from tqdm import tqdm
for key in  tqdm(range(len(anti_commuting_sets))):
# for key in anti_commuting_sets:
    if key not in xx.X_sk_Ops:
        for term in anti_commuting_sets[key]:
            PauliWord = term[0]
            constant = term[1]

            Paulis_to_be_tensored=[]
            for PauliString in PauliWord.split(' '):
                Op = OperatorsKeys[PauliString[0]]
                Paulis_to_be_tensored.append(Op)

            tensored_PauliWord = reduce(kron, Paulis_to_be_tensored)
            Operator += constant * tensored_PauliWord

    else:

        R_SL_DAGGER_matrix_LIST =[]
        R_SL_matrix_LIST = []
        for SL_terms in xx.X_sk_Ops[key]['X_sk_theta_sk']:
            theta_sk = SL_terms['theta_sk']
            X_sk = SL_terms['X_sk']
            constant = X_sk[1]

            X_sk_terms_to_be_tensored = [OperatorsKeys[PauliString[0]] for PauliString in X_sk[0].split(' ')]

            X_sk_matrix = reduce(kron, X_sk_terms_to_be_tensored)  #*X_sk[1] contains sign info!

            R_S_DAGGER_matrix = expm((-1j* theta_sk/2 * constant *X_sk_matrix)) #*X_sk[1] contains sign info!
            R_SL_DAGGER_matrix_LIST.append(R_S_DAGGER_matrix)

            R_S_matrix = expm((+1j * theta_sk/2 * constant * X_sk_matrix))
            R_SL_matrix_LIST.append(R_S_matrix)

        Pauliword_S = xx.X_sk_Ops[key]['PauliWord_S']
        beta_s = Pauliword_S[1]

        Paulis_to_be_tensored=[]
        for PauliString in Pauliword_S[0].split(' '):
            Op = OperatorsKeys[PauliString[0]]
            Paulis_to_be_tensored.append(Op)

        # next perform tensor!
        tensored_PauliWord = reduce(kron, Paulis_to_be_tensored)

        R_S_dagger = reduce(np.matmul, R_SL_DAGGER_matrix_LIST)
        R_S = reduce(np.matmul, R_SL_matrix_LIST)

        # Operator += xx.X_sk_Ops[key]['gamma_l'] * np.matmul(R_S, np.matmul(tensored_PauliWord, R_S_dagger)) # gamma * R_S P_S R_S_Dagger
        Operator += xx.X_sk_Ops[key]['gamma_l'] * beta_s * R_S * tensored_PauliWord * R_S_dagger  # gamma * R_S P_S R_S_Dagger


from scipy.sparse.linalg import eigs
eig_values, eig_vectors = eigs(Operator)
FCI_Energy = min(eig_values)
print(FCI_Energy)




P_words_and_consts=[]
for key in anti_commuting_sets:
    for term in anti_commuting_sets[key]:
        P_words_and_consts.append(term)

n_qubits = 4
Operator_standard = csr_matrix((2 ** n_qubits, 2 ** n_qubits))
from tqdm import tqdm
for i in tqdm(range(len(P_words_and_consts))):
    PauliWord = P_words_and_consts[i][0]
    constant = P_words_and_consts[i][1]

    Paulis_to_be_tensored = []
    for PauliString in PauliWord.split(' '):
        Op = OperatorsKeys[PauliString[0]]
        Paulis_to_be_tensored.append(Op)

    tensored_PauliWord = reduce(kron, Paulis_to_be_tensored)
    Operator_standard += constant * tensored_PauliWord

eig_values, eig_vectors = eigs(Operator_standard)
FCI_Energy = min(eig_values)
print(FCI_Energy)


