import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import jax.numpy as jnp
from jax import jit

class VectorPauliWord():

    def __init__(self, n_qubits, QubitOp):
        self.n_qubits = n_qubits
        self.QubitOp = QubitOp

        self.Pvec = None
        self._init_vector_form()

    def _init_vector_form(self):
        self.Pvec = jnp.zeros((2*self.n_qubits), dtype=int)

        PauliStr, const = tuple(*self.QubitOp.terms.items())
        for qNo, Pstr in PauliStr:
            if Pstr == 'X':
                self.Pvec=self.Pvec.at[qNo].set(1) # self.Pvec[qNo]=1

            if Pstr == 'Z':
                self.Pvec=self.Pvec.at[qNo+self.n_qubits].set(1)  # self.Pvec[qNo+self.n_qubits] = 1
        return None

def Commutes(VectorPauliWord_1, VectorPauliWord_2):

    # https://arxiv.org/pdf/1907.09386.pdf

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

class Vector_QubitHamiltonian():

    def __init__(self, QubitHamiltonian, n_qubits):
        self.QubitHamiltonian_list = list(QubitHamiltonian)
        self.n_qubits = n_qubits

        self.QubitHamiltonian_VectorPauliWord_list = []
        self._init_binary_matrix_form()


    def _init_binary_matrix_form(self):
        self.binary_mat= jnp.zeros((len(self.QubitHamiltonian_list), 2 * self.n_qubits), dtype=int) # (observabales, Paulivec_len)
        for ind, PauliOp in enumerate(self.QubitHamiltonian_list):
            VectorPWord = VectorPauliWord(self.n_qubits, PauliOp)
            self.binary_mat = self.binary_mat.at[ind,:].set(VectorPWord.Pvec)
            self.QubitHamiltonian_VectorPauliWord_list.append(VectorPWord)

        return None

    def Get_adj_mat(self, Pauli_grouping_type):

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
            J_symplectic = jnp.block([
                [jnp.zeros((self.n_qubits, self.n_qubits), dtype=int), jnp.eye(self.n_qubits, dtype=int)],
                [jnp.eye(self.n_qubits, dtype=int), jnp.zeros((self.n_qubits, self.n_qubits), dtype=int)]
                                      ],
                                    )

            vectorised_commuting_check = jnp.matmul(self.binary_mat,
                                              jnp.matmul(J_symplectic, jnp.transpose(self.binary_mat)))

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
        QubitHamiltonian:
        n_qubits:
        clique_relation:
        colouring_strategy:

    Returns:

    """


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


    return Cliques
