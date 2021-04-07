import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse import eye as sparse_eye
from scipy.sparse import bmat as sparse_block
from openfermion.ops import QubitOperator
from scipy.sparse import lil_matrix
from scipy.sparse import find


class VectorPauliWord():
    """
    Object of PauliWord that gives vector repesentation of Openfermion QubitOperator.
    Vector form first lists PauliX terms then PauliZ terms... as a ( 1 x 2 Nqubits)

    see pg8 of https://arxiv.org/pdf/1907.09386.pdf

    Args:
        n_qubits (int): Number of qubits
        QubitOp (Openfermion.ops.QubitOperator): Openfermion QubitOperator of 1 operator ONLY


    ** Example **

    >> from openfermion.ops import QubitOperator
    >> A = QubitOperator('X0 X1 X2', 1)
    >> Vec_A = VectorPauliWord(3, A)
    >> print(Vec_A.Pvec)

    output:
        [1 1 1 0 0 0]

    """

    def __init__(self, n_qubits, QubitOp):
        self.n_qubits = n_qubits
        self.QubitOp = QubitOp

        self.Pvec = None
        self._init_vector_form()

    def _init_vector_form(self):
        """
        Build vector from of PauliWord, as a ( 1 x 2 Nqubits) vector
        First N sites are Pauli X terms. Last N sites are Pauli Z terms
        """

        Pvec = np.zeros((2*self.n_qubits), dtype=int)

        PauliStr, const = tuple(*self.QubitOp.terms.items())
        for qNo, Pstr in PauliStr:
            if Pstr == 'X':
                Pvec[qNo] = 1

            elif Pstr == 'Z':
                Pvec[qNo+self.n_qubits] = 1

            elif Pstr == 'Y':
                Pvec[qNo] = 1
                Pvec[qNo+self.n_qubits] = 1
            else:
                raise ValueError(f'not Pauli operator: {Pstr} on qubit {qNo}')
        
        self.Pvec = csr_matrix(Pvec, shape=(1, 2*self.n_qubits), dtype=int)
        return None

def Commutes(VectorPauliWord_1, VectorPauliWord_2):
    """

    See https://arxiv.org/pdf/1907.09386.pdf for mathematical background of symplectic approach.

    Args:
        VectorPauliWord_1 (VectorPauliWord): vector object of PauliWord
        VectorPauliWord_2 (VectorPauliWord): vector object of other PauliWord

    Returns:
        bool: True if operators commute else returns False not

    """


    if VectorPauliWord_1.n_qubits!=VectorPauliWord_2.n_qubits:
        raise ValueError('qubit number mismatch')

    n_qubits = VectorPauliWord_1.n_qubits
    sparse_zeros = csr_matrix(np.zeros((n_qubits,n_qubits), dtype=int))
    J_symplectic = sparse_block([
                                [sparse_zeros , sparse_eye(n_qubits, dtype=int)],
                                [sparse_eye(n_qubits), sparse_zeros]
                            ])

    commutation = VectorPauliWord_1.Pvec.dot(J_symplectic.dot(VectorPauliWord_2.Pvec.T))

    if commutation.toarray()%2 == 0:
        return True
    else:
        return False

def QWC(VectorPauliWord_1, VectorPauliWord_2):
    """

    Cannot use symplectic approach. Iteratively goes through each vectorised Pauliword, checking if
     they all qubit-wise commute (QWC)

    Args:
        VectorPauliWord_1 (VectorPauliWord): vector object of PauliWord
        VectorPauliWord_2 (VectorPauliWord): vector object of other PauliWord

    Returns:
        bool: True if operators QWC else returns False not

    """
    if VectorPauliWord_1.n_qubits != VectorPauliWord_2.n_qubits:
        raise ValueError('qubit number mismatch')

    n_qubits = VectorPauliWord_1.n_qubits
    for qbit in range(n_qubits):

        qbit_vec1 = (VectorPauliWord_1.Pvec[0, qbit], VectorPauliWord_1.Pvec[0, qbit+n_qubits]) # (X_vec, Z_vec)

        if qbit_vec1 == (0,0):
            # identity on qubit of Vec1
            continue

        qbit_vec2 = (VectorPauliWord_2.Pvec[0, qbit],  VectorPauliWord_2.Pvec[0, qbit+n_qubits])  # (X_vec, Z_vec)

        if qbit_vec2 == (0,0):
            # identity on qubit of Vec2
            continue

        if qbit_vec1 != qbit_vec2:
            # different operator on qubit
            return False

    return True

def vectorised_full_commuting_check(n_qubits, binary_H_mat):

    sparse_zeros = csr_matrix(np.zeros((n_qubits,n_qubits), dtype=int))
    J_symplectic = sparse_block([
                                [sparse_zeros , sparse_eye(n_qubits, dtype=int)],
                                [sparse_eye(n_qubits), sparse_zeros]
                            ])

    vectorised_commuting_check = binary_H_mat.dot(J_symplectic.dot(np.transpose(binary_H_mat)))
    return vectorised_commuting_check

class Vector_QubitHamiltonian():
    """
    Build matrix form of QubitHamiltonian. Allows fast way to get Hamiltonian graph adjacency matrix

    Args:
        n_qubits (int): Number of qubits
        QubitHamiltonian (Openfermion.ops.QubitOperator): Openfermion QubitOperator of 1 or MORE operator


    ** Example **

    >> from openfermion.ops import QubitOperator
    >> n_qubits = 3
    >> H = QubitOperator('X0 X1 X2') + QubitOperator('X1 X2') + QubitOperator('Z0 Z2') + QubitOperator('X2')
    >> H_vec = Vector_QubitHamiltonian(H, n_qubits)
    >> adj_mat = H_vec.Get_adj_mat('C')

    >> print([P.QubitOp for P in H_vec.QubitHamiltonian_VectorPauliWord_list])
    >> print(adj_mat)

    output:
        [   1.0 [X0 X1 X2],
            1.0 [X1 X2],
            1.0 [Z0 Z2],
            1.0 [X2]
        ]

        [[0 1 1 1]
         [1 0 0 1]
         [1 0 0 0]
         [1 1 0 0]]

    """
    def __init__(self, QubitHamiltonian, n_qubits):
        self.QubitHamiltonian_list = list(QubitHamiltonian)
        self.n_qubits = n_qubits

        self.Qubit_H_vec_ind_to_Pvec = {}
        self._init_binary_matrix_form()


    def _init_binary_matrix_form(self):
        """
        Turn QubitHamiltonian of M terms into an (M * 2N_qubits) matrix. Each row represents a Pauli Operator.
        Useful as can find commutativity relationships in a vectorised way.
        """

        # binary_mat= np.zeros((len(self.QubitHamiltonian_list), 2 * self.n_qubits), dtype=int) # (observabales, Paulivec_len)
        # for ind, PauliOp in enumerate(self.QubitHamiltonian_list):
        #     VectorPWord = VectorPauliWord(self.n_qubits, PauliOp)
        #     binary_mat[ind,:] = VectorPWord.Pvec.toarray()
        #     self.QubitHamiltonian_VectorPauliWord_list.append(VectorPWord)

        # self.binary_mat = csr_matrix(binary_mat, dtype=int)

        # self.binary_mat= csr_matrix(np.zeros((len(self.QubitHamiltonian_list),2 * self.n_qubits)), shape=(len(self.QubitHamiltonian_list),2 * self.n_qubits), dtype=int)# (observabales, Paulivec_len)
        self.binary_mat= lil_matrix((len(self.QubitHamiltonian_list),2 * self.n_qubits), dtype=int)
        for ind, PauliOp in enumerate(self.QubitHamiltonian_list):
            VectorPWord = VectorPauliWord(self.n_qubits, PauliOp)
            self.binary_mat[ind,:] = VectorPWord.Pvec
            self.Qubit_H_vec_ind_to_Pvec[ind] = VectorPWord
        return None

    def Get_adj_mat(self, Pauli_grouping_type):
        """


        Build the adjacency matrix for an undirected graph of M nodes - each node reprsenting a term in the Hamiltonian.
        The adjacency matrix is a M by M symmetric binary matrix, where matrix elements of 1 denote an edge, and
        matrix elements of 0 denote no edge. To get term of Hamiltonian from index use the Qubit_H_vec_ind_to_Pvec dictionary

        Args:
            Pauli_grouping_type (str): Commuting relationship between node edges
                                       (C = commuting, AC = anticommuting, QWC = qubitwise commuting)

        Returns:
            adjacency_mat (scipy.sparse array): matrix repsenting which terms are connected by edge
                                                              representing commuting relationship met.
        """
        if (Pauli_grouping_type == 'QWC') or (Pauli_grouping_type=='qubit-wise commuting'):

            adjacency_mat = np.zeros((self.binary_mat.shape[0], self.binary_mat.shape[0]), dtype=int)

            for P_vec_row_ind in range(self.binary_mat.shape[0]):
                for j in range(P_vec_row_ind+1, self.binary_mat.shape[0]): # only loop over bottom triangle

                    does_qwc = int(QWC(self.Qubit_H_vec_ind_to_Pvec[P_vec_row_ind],
                                       self.Qubit_H_vec_ind_to_Pvec[j])
                                   )

                    adjacency_mat[(P_vec_row_ind, j)] =does_qwc #[i, j]
                    adjacency_mat[(j, P_vec_row_ind)] =does_qwc # [j,i]

        else:

            vectorised_commuting_check = vectorised_full_commuting_check(self.n_qubits, self.binary_mat)

            if (Pauli_grouping_type == 'AC') or (Pauli_grouping_type=='anitcommuting'):
                adjacency_mat = vectorised_commuting_check.toarray() % 2

                ## TODO: below approach may be faster... as this does NOT make full dense array
                # row, col, values = find(Hamilt_vector.binary_mat)
                # modulo_values = values%2
                # adjacency_mat= csr_matrix((modulo_values, (row, col)), shape=(Hamilt_vector.binary_mat.shape[0], Hamilt_vector.binary_mat.shape[0]))

            elif (Pauli_grouping_type == 'C') or (Pauli_grouping_type=='commuting'):
                adjacency_mat = (vectorised_commuting_check.toarray()+1) % 2
                np.fill_diagonal(adjacency_mat, 0)

            else:
                raise ValueError(f'Unknown Pauli_grouping_type {Pauli_grouping_type}')

        return adjacency_mat #csr_matrix(adjacency_mat)

def Clique_cover_Hamiltonian(QubitHamiltonian, n_qubits, clique_relation, colouring_strategy, colour_interchange=False):

    """
    Find Clique cover of a given Openfermion QubitOperator representing a Hamiltonian

    Note for colour interchange
    https://fileadmin.cs.lth.se/cs/Personal/Andrzej_Lingas/k-m.pdf

    Args:
        QubitHamiltonian (Openfermion.ops.QubitOperator): Openfermion QubitOperator of 1 or MORE operator
        n_qubits: Total number of qubits for system
        clique_relation: Commuting relationship between terms in Hamiltonian
        colouring_strategy: Graph Colouring strategy
        colour_interchange (bool): Flag to perform colour interchange alg

    Returns:
        Cliques (dict): Dictionary of cliques following specified clique_relation


    ** Example **

    >> from openfermion.ops import QubitOperator
    >> n_qubits = 3
    >> H = QubitOperator('X0 X1 X2') + QubitOperator('X1 X2') + QubitOperator('Z0 Z2') + QubitOperator('X2')
    >> CliqueCover = Clique_cover_Hamiltonian(H, n_qubits, 'AC', 'largest_first')
    >> print(CliqueCover)

    output:
        {
             0: [1.0 [X0 X1 X2]],
             1: [1.0 [X1 X2], 1.0 [Z0 Z2]],
             2: [1.0 [X2]]
        }
    """

    if isinstance(QubitHamiltonian, list):
        H = QubitOperator()
        for Q_op in QubitHamiltonian: H+=Q_op
        QubitHamiltonian=H

    Hamilt_vector = Vector_QubitHamiltonian(QubitHamiltonian, n_qubits)

    adj_matrix = Hamilt_vector.Get_adj_mat(clique_relation)
    node_conv_dict = {ind: Pvec.QubitOp for ind, Pvec in Hamilt_vector.Qubit_H_vec_ind_to_Pvec.items()}

    Graph = nx.from_numpy_matrix(adj_matrix)
    Complement_Graph = nx.complement(Graph)

    greedy_colouring_output_dic = nx.greedy_color(Complement_Graph, strategy=colouring_strategy, interchange=colour_interchange)
    unique_colours = set(greedy_colouring_output_dic.values())

    Cliques = {}
    for Clique_ind in unique_colours:
        Cliques[Clique_ind] = [node_conv_dict[k] for k in greedy_colouring_output_dic.keys()
                                        if greedy_colouring_output_dic[k] == Clique_ind]


    # colour_key_for_nodes = {}
    # for colour in unique_colours:
    #     colour_key_for_nodes[colour] = [k for k in greedy_colouring_output_dic.keys()
    #                                     if greedy_colouring_output_dic[k] == colour]

    # if plot_graph is True:
    #     import matplotlib.cm as cm
    #     colour_list = cm.rainbow(np.linspace(0, 1, len(Cliques)))
    #     pos = nx.circular_layout(Graph)
    #
    #     for colour_ind, clique_list in enumerate(Cliques.values()):
    #         nx.draw_networkx_nodes(Graph, pos,
    #                                nodelist=[node_ind for node_ind in range(len(clique_list))],
    #                                node_color=colour_list[colour_ind].reshape([1, 4]),
    #                                node_size=500,
    #                                alpha=0.8)
    #
    #     labels = {node: clique_P_ops for node, clique_P_ops in enumerate(clique_list)}
    #     seperator = ' '
    #     labels = {node: seperator.join([tup[1] + str(tup[0]) for tup in node[0]]) for node in list(Graph.nodes)}
    #     #
    #     # nx.draw_networkx_labels(Graph, pos, labels)  # , font_size=8)
    #
    #     nx.draw_networkx_edges(Graph, pos, width=1.0, alpha=0.5)
    #
    #     # plt.savefig('coloured_G', dpi=300, transparent=True, )  # edgecolor='black', facecolor='white')
    #
    #     plt.show()


    del Hamilt_vector
    del Graph
    del Complement_Graph
    return Cliques