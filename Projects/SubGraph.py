from quchem.Hamiltonian_Generator_Functions import *

### Variable Parameters
Molecule = 'LiH'#LiH'
geometry = None
num_shots = 10000
HF_occ_index = [0,1,2] #[0, 1,2] # for occupied_orbitals_index_list
#######

### Get Hamiltonian
Hamilt = Hamiltonian(Molecule,
                     run_scf=1, run_mp2=1, run_cisd=1, run_ccsd=1, run_fci=1,
                     basis='sto-3g',
                     multiplicity=1,
                     geometry=geometry)  # normally None!

Hamilt.Get_Molecular_Hamiltonian()
SQ_CC_ops, THETA_params = Hamilt.Get_ia_and_ijab_terms(Coupled_cluser_param=True)
#print('UCC operations: ', SQ_CC_ops)

HF_transformations = Hamiltonian_Transforms(Hamilt.MolecularHamiltonian, SQ_CC_ops, Hamilt.molecule.n_qubits)

QubitHam = HF_transformations.Get_Qubit_Hamiltonian_JW()
#print('Qubit Hamiltonian: ', QubitHam)
QubitHam_PauliStr = HF_transformations.Convert_QubitMolecularHamiltonian_To_Pauliword_Str_list(QubitHam)
#print('Qubit Hamiltonian: ', QubitHam_PauliStr)

## calc energy via Lin. Alg.
# UCC_JW_excitation_matrix_list = HF_transformations.Get_Jordan_Wigner_CC_Matrices()
# HF_ref_ket, HF_ref_bra = Hamilt.Get_Basis_state_in_occ_num_basis(occupied_orbitals_index_list=HF_occ_index)
# w = CalcEnergy(Hamilt.MolecularHamiltonianMatrix, HF_ref_ket, Hamilt.molecule.n_qubits,
#                UCC_JW_excitation_matrix_list)
# w.Calc_HF_Energy()
# w.Calc_UCCSD_No_Trot(THETA_params)
# w.Calc_UCCSD_with_Trot(THETA_params)

### Ansatz ###
# from quchem.Ansatz_Generator_Functions import *
#
# UCCSD = UCCSD_Trotter(SQ_CC_ops, THETA_params)
# Second_Quant_CC_JW_OP_list = UCCSD.SingleTrotterStep()
# PauliWord_list = Convert_QubitOperator_To_Pauliword_Str_list(Second_Quant_CC_JW_OP_list)
# HF_UCCSD_ansatz = Ansatz_Circuit(PauliWord_list, Hamilt.molecule.n_electrons, Hamilt.molecule.n_qubits)
# # THETA_params = [random.uniform(0, 2 * np.pi) for _ in range(Hamilt.num_theta_parameters)]
# ansatz_Q_cicuit = HF_UCCSD_ansatz.Get_Full_HF_UCCSD_QC(THETA_params)
# print(ansatz_Q_cicuit)
#


### Graph Colouring
from quchem.Graph import *

List_PauliWords, HamiltonainCofactors = zip(*QubitHam_PauliStr)

attribute_dictionary = {'Cofactors': HamiltonainCofactors}

List_of_nodes, node_attributes_dict = Get_list_of_nodes_and_attributes(List_PauliWords,
                                                                       attribute_dictionary=attribute_dictionary)

G =  Hamiltonian_Graph(List_PauliWords, Graph_colouring_strategy='largest_first', attribute_dictionary=attribute_dictionary)
anti_commuting_sets = G.Get_Pauli_grouping('C', plot_graph=False)


anti_commuting_set_stripped = Get_PauliWord_constant_tuples(anti_commuting_sets, dict_str_label='Cofactors')
print(anti_commuting_set_stripped)

# GET SUBGRAPH of graph:

GRAPH = nx.Graph()
for key in anti_commuting_set_stripped:
    GRAPH.add_node(key)

for key in anti_commuting_set_stripped:
    set1_P, set1_C = zip(*anti_commuting_set_stripped[key])

    for k in range(key+1, len(anti_commuting_set_stripped)):
        set2_P, set2_C = zip(*anti_commuting_set_stripped[k])

        Graph_of_sets = Graph_of_two_sets(set1_P, set2_P,
                                                  'C', plot_graph=False, node_attributes_dict=None)

        if Check_if_sets_completely_connected(Graph_of_sets, set1_P, set2_P):
            GRAPH.add_edge(key, k)

# list(GRAPH.edges)



single_G, multi_G = Get_subgraphs(GRAPH, node_attributes_dict=None)
s_colour = Colour_list_of_Graph(GRAPH, attribute_dictionary=None, plot_graph=False,
                                strategy='largest_first')
m_colour = Colour_list_of_Graph(GRAPH, attribute_dictionary=None, plot_graph=False,
                                strategy='largest_first')

output_sets = Get_unique_graph_colours(s_colour + m_colour)


def Get_subgraphs(Graph):

    # -*- coding: utf-8 -*-
    #    Copyright (C) 2004-2016 by
    #    Aric Hagberg <hagberg@lanl.gov>
    #    Dan Schult <dschult@colgate.edu>
    #    Pieter Swart <swart@lanl.gov>
    #    All rights reserved.
    #    BSD license.
    #
    # Authors: Eben Kenah
    #          Aric Hagberg (hagberg@lanl.gov)
    #          Christopher Ellison
    """Connected components."""

    if isinstance(Graph, nx.classes.digraph.DiGraph):
        raise TypeError('Cannot have a directed graph, must be a undirected graph')

    def connected_components(G):
        """Generate connected components.

        Parameters
        ----------
        G : NetworkX graph
           An undirected graph

        Returns
        -------
        comp : generator of sets
           A generator of sets of nodes, one for each component of G.

        Raises
        ------
        NetworkXNotImplemented:
            If G is undirected.

        Examples
        --------
        Generate a sorted list of connected components, largest first.

        # >>> G = nx.path_graph(4)
        # >>> nx.add_path(G, [10, 11, 12])
        # >>> [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
        [4, 3]

        If you only want the largest connected component, it's more
        efficient to use max instead of sort.

        # >>> largest_cc = max(nx.connected_components(G), key=len)

        See Also
        --------
        strongly_connected_components
        weakly_connected_components

        Notes
        -----
        For undirected graphs only.

        """
        seen = set()
        for v in G:
            if v not in seen:
                c = set(_plain_bfs(G, v))
                yield c
                seen.update(c)
    def connected_component_subgraphs(G, copy=True):
        """Generate connected components as subgraphs.

        Parameters
        ----------
        G : NetworkX graph
           An undirected graph.

        copy: bool (default=True)
          If True make a copy of the graph attributes

        Returns
        -------
        comp : generator
          A generator of graphs, one for each connected component of G.

        Raises
        ------
        NetworkXNotImplemented:
            If G is undirected.

        Examples
        --------
        # >>> G = nx.path_graph(4)
        # >>> G.add_edge(5,6)
        # >>> graphs = list(nx.connected_component_subgraphs(G))

        If you only want the largest connected component, it's more
        efficient to use max instead of sort:

        # >>> Gc = max(nx.connected_component_subgraphs(G), key=len)

        See Also
        --------
        connected_components
        strongly_connected_component_subgraphs
        weakly_connected_component_subgraphs

        Notes
        -----
        For undirected graphs only.
        Graph, node, and edge attributes are copied to the subgraphs by default.

        """
        for c in connected_components(G):
            if copy:
                yield G.subgraph(c).copy()
            else:
                yield G.subgraph(c)
    def _plain_bfs(G, source):
        """A fast BFS node generator"""
        seen = set()
        nextlevel = {source}
        while nextlevel:
            thislevel = nextlevel
            nextlevel = set()
            for v in thislevel:
                if v not in seen:
                    yield v
                    seen.add(v)
                    nextlevel.update(G[v])

    connected_graphs = list(connected_component_subgraphs(Graph))
    multi_node_G = []
    single_node_G = []
    for graph in connected_graphs:
        if len(graph.nodes) > 1:
            multi_node_G.append(graph)
        else:
            single_node_G.append(graph)

    return single_node_G, multi_node_G

def Colour_list_of_Graph(Graph_list, plot_graph=False, strategy='largest_first'):
    # different strategies at:
    # https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.coloring.greedy_color.html#networkx.algorithms.coloring.greedy_color

    List_of_Coloured_Graphs_dicts = []
    for graph in Graph_list:
        greedy_colouring_output_dic = nx.greedy_color(graph, strategy=strategy, interchange=False)
        unique_colours = set(greedy_colouring_output_dic.values())

        colour_key_for_nodes = {}
        for colour in unique_colours:
            if attribute_dictionary is None:
                colour_key_for_nodes[colour] = [k for k in greedy_colouring_output_dic.keys()
                                                       if greedy_colouring_output_dic[k] == colour]

        List_of_Coloured_Graphs_dicts.append(colour_key_for_nodes)

        if plot_graph == True:
            import matplotlib.cm as cm
            plt.figure()
            colour_list = cm.rainbow(np.linspace(0, 1, len(colour_key_for_nodes)))
            pos = nx.circular_layout(graph)
            for colour in colour_key_for_nodes:
                nx.draw_networkx_nodes(graph, pos,
                                       nodelist=[P_word for i in colour_key_for_nodes[colour]
                                                 for P_word, const in i.items()],
                                       node_color=colour_list[colour],
                                       node_size=500,
                                       alpha=0.8
                                     )
                labels = {P_word: P_word for i in colour_key_for_nodes[colour] for P_word, const in i.items()}
                nx.draw_networkx_labels(graph, pos, labels)  # , font_size=8)

            nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
            plt.show()
    return List_of_Coloured_Graphs_dicts