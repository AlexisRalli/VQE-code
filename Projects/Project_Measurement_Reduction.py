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
print('UCC operations: ', SQ_CC_ops)

HF_transformations = Hamiltonian_Transforms(Hamilt.MolecularHamiltonian, SQ_CC_ops, Hamilt.molecule.n_qubits)

QubitHam = HF_transformations.Get_Qubit_Hamiltonian_JW()
#print('Qubit Hamiltonian: ', QubitHam)
QubitHam_PauliStr = HF_transformations.Convert_QubitMolecularHamiltonian_To_Pauliword_Str_list(QubitHam)
print('Qubit Hamiltonian: ', QubitHam_PauliStr)

## calc energy via Lin. Alg.
# UCC_JW_excitation_matrix_list = HF_transformations.Get_Jordan_Wigner_CC_Matrices()
# HF_ref_ket, HF_ref_bra = Hamilt.Get_Basis_state_in_occ_num_basis(occupied_orbitals_index_list=HF_occ_index)
# w = CalcEnergy(Hamilt.MolecularHamiltonianMatrix, HF_ref_ket, Hamilt.molecule.n_qubits,
#                UCC_JW_excitation_matrix_list)
# w.Calc_HF_Energy()
# w.Calc_UCCSD_No_Trot(THETA_params)
# w.Calc_UCCSD_with_Trot(THETA_params)

### Ansatz ###
from quchem.Ansatz_Generator_Functions import *

UCCSD = UCCSD_Trotter(SQ_CC_ops, THETA_params)
Second_Quant_CC_JW_OP_list = UCCSD.SingleTrotterStep()
PauliWord_list = Convert_QubitOperator_To_Pauliword_Str_list(Second_Quant_CC_JW_OP_list)
HF_UCCSD_ansatz = Ansatz_Circuit(PauliWord_list, Hamilt.molecule.n_electrons, Hamilt.molecule.n_qubits)
# THETA_params = [random.uniform(0, 2 * np.pi) for _ in range(Hamilt.num_theta_parameters)]
ansatz_Q_cicuit = HF_UCCSD_ansatz.Get_Full_HF_UCCSD_QC(THETA_params)
print(ansatz_Q_cicuit)



### Graph Colouring
from quchem.Graph import *

List_PauliWords, HamiltonainCofactors = zip(*QubitHam_PauliStr)

attribute_dictionary = {'Cofactors': HamiltonainCofactors}

List_of_nodes, node_attributes_dict = Get_list_of_nodes_and_attributes(List_PauliWords,
                                                                       attribute_dictionary=attribute_dictionary)

G = nx.Graph()
G = Build_Graph_Nodes(List_of_nodes, G, node_attributes_dict=node_attributes_dict, plot_graph=False)
G = Build_Graph_Edges_COMMUTING_QWC_AntiCommuting(G, List_of_nodes, 'C', plot_graph=False)

# comp_G = Get_Complemenary_Graph(G, node_attributes_dict=node_attributes_dict, plot_graph=True) # <- not currently used

single_G, multi_G = Get_subgraphs(G, node_attributes_dict=node_attributes_dict)
s_colour = Colour_list_of_Graph(single_G, attribute_dictionary=attribute_dictionary, plot_graph=False,
                                strategy='largest_first')
m_colour = Colour_list_of_Graph(multi_G, attribute_dictionary=attribute_dictionary, plot_graph=False,
                                strategy='largest_first')

anti_commuting_sets = Get_unique_graph_colours(s_colour + m_colour)

anti_commuting_set_stripped = Get_PauliWord_constant_tuples(anti_commuting_sets, dict_str_label='Cofactors')

print(anti_commuting_set_stripped)

### NEXT graph!
set1_P, set1_C = zip(*anti_commuting_set_stripped[6])
set2_P, set2_C = zip(*anti_commuting_set_stripped[8])

NEW_attribute_dictionary = {'Cofactors': set1_C+set2_C}


List_of_nodes, node_attributes_dict = Get_list_of_nodes_and_attributes(set1_P+set2_P,
                                                                       attribute_dictionary=NEW_attribute_dictionary)

G = nx.Graph()
G = Build_Graph_Nodes(List_of_nodes, G, node_attributes_dict=node_attributes_dict, plot_graph=False)
G = Build_Graph_Edges_COMMUTING_QWC_AntiCommuting(G, List_of_nodes, 'AC', plot_graph=False)

single_G, multi_G = Get_subgraphs(G, node_attributes_dict=node_attributes_dict)
s_colour = Colour_list_of_Graph(single_G, attribute_dictionary=attribute_dictionary, plot_graph=False,
                                strategy='largest_first')
m_colour = Colour_list_of_Graph(multi_G, attribute_dictionary=attribute_dictionary, plot_graph=True,
                                strategy='largest_first')

commuting_sets = Get_unique_graph_colours(s_colour + m_colour)


key_i=6
term_i=0
key_j=8
term_j=0

P1 =list(commuting_sets[i][term_i].keys())[0]
P2 =list(commuting_sets[j][term_j].keys())[0]
print(Commutativity(P1, P2, 'C'))


def Graph_of_two_sets(Graph, PauliWord_string_nodes_list_1, PauliWord_string_nodes_list_2,
                                                  anti_comm_QWC, plot_graph = False):
    """

    Function builds graph edges for commuting / anticommuting / QWC PauliWords

    Args:
        PauliWord_string_nodes_list (list): list of PauliWords (str)
        Graph: networkX graph with nodes already defined
        anti_comm_QWC (str): flags to find either:
                                           qubit wise commuting (QWC) terms  -> flag = 'QWC',
                                                             commuting terms -> flag = 'C',
                                                        anti-commuting terms -> flag = 'AC'
        plot_graph (optional, bool): whether to plot graph

    Returns:
        Graph: Graph with nodes connected if they commute / QWC / anti-commute

    """

    # Build nodes
    labels={}
    for node in [*PauliWord_string_nodes_list_1, *PauliWord_string_nodes_list_2]:
        Graph.add_node(node)
        labels[node] = node

    pos = nx.circular_layout(Graph)

    nx.draw_networkx_nodes(Graph, pos,
                           nodelist=PauliWord_string_nodes_list_1,
                           node_color='r',
                           node_size=500,
                           alpha=0.8)
    nx.draw_networkx_nodes(Graph, pos,
                           nodelist=PauliWord_string_nodes_list_2,
                           node_color='b',
                           node_size=500,
                           alpha=0.8)

    nx.draw_networkx_labels(Graph, pos, labels)  # , font_size=8)

    if node_attributes_dict is not None:
        nx.set_node_attributes(Graph, node_attributes_dict)


    # Build Edges
    edgelist=[]
    for i in tqdm(range(len(PauliWord_string_nodes_list_1)), ascii=True, desc='Building Graph Edges'):
        selected_PauliWord = PauliWord_string_nodes_list_1[i]

        for comparison_PauliWord in PauliWord_string_nodes_list_2:

            if Commutativity(selected_PauliWord, comparison_PauliWord, anti_comm_QWC) is True:
                Graph.add_edge(selected_PauliWord, comparison_PauliWord)
                edgelist.append((selected_PauliWord,comparison_PauliWord))
            else:
                continue
    nx.draw_networkx_edges(Graph, pos,
                           edgelist=edgelist,
                           width=2, alpha=0.5, edge_color='k')

    if plot_graph == True:
        plt.figure()
        plt.show()
    return Graph

H = nx.Graph()
anti_comm_QWC = 'C'
H = Graph_of_two_sets(H, set1_P, set2_P, anti_comm_QWC, plot_graph = True)