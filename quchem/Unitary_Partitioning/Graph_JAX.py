import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import jax.numpy as jnp
from jax import jit
from openfermion.ops import QubitOperator


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

        self.Pvec = jnp.zeros((2*self.n_qubits), dtype=int)

        PauliStr, const = tuple(*self.QubitOp.terms.items())
        for qNo, Pstr in PauliStr:
            if Pstr == 'X':
                self.Pvec=self.Pvec.at[qNo].set(1) # self.Pvec[qNo]=1

            elif Pstr == 'Z':
                self.Pvec=self.Pvec.at[qNo+self.n_qubits].set(1)  # self.Pvec[qNo+self.n_qubits] = 1


            elif Pstr == 'Y':
                self.Pvec=self.Pvec.at[qNo].set(1) # self.Pvec[qNo]=1
                self.Pvec=self.Pvec.at[qNo+self.n_qubits].set(1)  # self.Pvec[qNo+self.n_qubits] = 1
            else:
                raise ValueError(f'not Pauli operator: {Pstr} on qubit {qNo}')
                
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
    J_symplectic = jnp.block([
                                [jnp.zeros((n_qubits,n_qubits)) ,jnp.eye(n_qubits)],
                                [jnp.eye(n_qubits), jnp.zeros((n_qubits,n_qubits))]
                            ],
                           )

    commutation = jnp.dot(VectorPauliWord_1.Pvec, jnp.matmul(J_symplectic, VectorPauliWord_2.Pvec))

    if commutation%2 == 0:
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

        qbit_vec1 = (VectorPauliWord_1.Pvec[qbit], VectorPauliWord_1.Pvec[qbit+n_qubits]) # (X_vec, Z_vec)

        if qbit_vec1 == (0,0):
            # identity on qubit of Vec1
            continue

        qbit_vec2 = (VectorPauliWord_2.Pvec[qbit],  VectorPauliWord_2.Pvec[qbit+n_qubits])  # (X_vec, Z_vec)

        if qbit_vec2 == (0,0):
            # identity on qubit of Vec2
            continue

        if qbit_vec1 != qbit_vec2:
            # different operator on qubit
            return False

    return True


def vectorised_full_commuting_check(n_qubits, binary_H_mat):
    J_symplectic = jnp.block([
                            [jnp.zeros((n_qubits, n_qubits), dtype=int), jnp.eye(n_qubits, dtype=int)],
                            [jnp.eye(n_qubits, dtype=int), jnp.zeros((n_qubits, n_qubits), dtype=int)]
                                                  ],
                                                )

    vectorised_commuting_check = jnp.matmul(binary_H_mat,
                                  jnp.matmul(J_symplectic, jnp.transpose(binary_H_mat))) 
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

        self.QubitHamiltonian_VectorPauliWord_list = []
        self._init_binary_matrix_form()


    def _init_binary_matrix_form(self):
        """
        Turn QubitHamiltonian of M terms into an (M * 2N_qubits) matrix. Each row represents a Pauli Operator.
        Useful as can find commutativity relationships in a vectorised way.
        """

        self.binary_mat= jnp.zeros((len(self.QubitHamiltonian_list), 2 * self.n_qubits), dtype=int) # (observabales, Paulivec_len)
        for ind, PauliOp in enumerate(self.QubitHamiltonian_list):
            VectorPWord = VectorPauliWord(self.n_qubits, PauliOp)
            self.binary_mat = self.binary_mat.at[ind,:].set(VectorPWord.Pvec)
            self.QubitHamiltonian_VectorPauliWord_list.append(VectorPWord)

        return None

    def Get_adj_mat(self, Pauli_grouping_type):
        """

        Args:
            Pauli_grouping_type (str): Commuting relationship between node edges
                                       (C = commuting, AC = anticommuting, QWC = qubitwise commuting)

        Returns:
            adjacency_mat (jaxlib.xla_extension.DeviceArray): matrix repsenting which terms are connected by edge
                                                              representing commuting relationship met.
        """
        if (Pauli_grouping_type == 'QWC') or (Pauli_grouping_type=='qubit-wise commuting'):

            adjacency_mat = jnp.zeros((self.binary_mat.shape[0], self.binary_mat.shape[0]), dtype=int)

            for P_vec_row_ind in range(self.binary_mat.shape[0]):
                for j in range(P_vec_row_ind+1, self.binary_mat.shape[0]): # only loop over bottom triangle

                    does_qwc = int(QWC(self.QubitHamiltonian_VectorPauliWord_list[P_vec_row_ind],
                                       self.QubitHamiltonian_VectorPauliWord_list[j])
                                   )

                    adjacency_mat = adjacency_mat.at[(P_vec_row_ind, j)].set(does_qwc) #[i, j]
                    adjacency_mat = adjacency_mat.at[(j, P_vec_row_ind)].set(does_qwc) # [j,i]

        else:
            # J_symplectic = jnp.block([
            #     [jnp.zeros((self.n_qubits, self.n_qubits), dtype=int), jnp.eye(self.n_qubits, dtype=int)],
            #     [jnp.eye(self.n_qubits, dtype=int), jnp.zeros((self.n_qubits, self.n_qubits), dtype=int)]
            #                           ],
            #                         )

            # vectorised_commuting_check = jnp.matmul(self.binary_mat,
            #                                   jnp.matmul(J_symplectic, jnp.transpose(self.binary_mat)).block_until_ready()).block_until_ready() 

            fast_vectorised_comm_check_fn = jit(vectorised_full_commuting_check)
            vectorised_commuting_check = fast_vectorised_comm_check_fn(self.n_qubits, self.binary_mat).block_until_ready()

            if (Pauli_grouping_type == 'AC') or (Pauli_grouping_type=='anitcommuting'):
                adjacency_mat = vectorised_commuting_check % 2

            elif (Pauli_grouping_type == 'C') or (Pauli_grouping_type=='commuting'):
                adjacency_mat = (vectorised_commuting_check+1) % 2

                # set terms on diagonal to zero
                diag_ind = jnp.diag_indices_from(adjacency_mat)
                adjacency_mat = adjacency_mat.at[diag_ind].set(0)

            else:
                raise ValueError(f'Unknown Pauli_grouping_type {Pauli_grouping_type}')

        return adjacency_mat

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
    node_conv_dict = {ind: P.QubitOp for ind, P in enumerate( Hamilt_vector.QubitHamiltonian_VectorPauliWord_list)}
    Graph = nx.from_numpy_matrix(adj_matrix)
    Complement_Graph = nx.complement(Graph)

    greedy_colouring_output_dic = nx.greedy_color(Complement_Graph, strategy=colouring_strategy, interchange=colour_interchange)
    unique_colours = set(greedy_colouring_output_dic.values())

    Cliques = {}
    for Clique_ind in unique_colours:
        Cliques[Clique_ind] = [node_conv_dict[k] for k in greedy_colouring_output_dic.keys()
                                        if greedy_colouring_output_dic[k] == Clique_ind]

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
