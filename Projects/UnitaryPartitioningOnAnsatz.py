#from quchem.Ansatz_Generator_Functions import HF_state_generator
from quchem.Ansatz_Generator_Functions import *
from quchem.Graph import *


n_electrons = 3
n_qubits = 12

HF_initial_state= HF_state_generator(n_electrons, n_qubits)

UCC = Full_state_prep_circuit(HF_initial_state, T1_and_T2_theta_list=[])
UCC.complete_UCC_circuit()
#print(UCC.UCC_full_circuit)
print(UCC.T1_formatted)
print(UCC.T2_formatted)


# for T_Tdag in UCC.T1_formatted:
#     factor = len(T_Tdag)
#     for P_word, const in T_Tdag:
#         print(P_word)
# Commutativity(UCC.T1_formatted[0][0][0], UCC.T1_formatted[0][1][0], 'C')


List_of_P_Words_RAW = [P_word for T_Tdag in UCC.T2_formatted for P_word, const in T_Tdag]

longest_term = max(List_of_nodes, key=len)
max_qubit =int(longest_term.split(' ')[-1][1:])
PauliWords_list = []
for PauliWord in List_of_P_Words_RAW:
    if len(PauliWord) < len(longest_term):
        up_to = int(PauliWord.split(' ')[-1][1:])
        missing_terms = ['I{}'.format(i)for i in np.arange(up_to +1, max_qubit +1, 1)]

        seperator = ' '
        missing_terms = seperator.join(missing_terms)
        PauliWords_list.append(PauliWord + ' ' + missing_terms)
    else:
        PauliWords_list.append(PauliWord)
List_of_nodes = PauliWords_list

key=0
UCC_terms={}
for T_Tdag in UCC.T2_formatted:
    factor = len(T_Tdag)
    UCC_terms[key] = T_Tdag
    key+=1


INDEX = 0
List_of_nodes = [P_word_const_tuple[0] for P_word_const_tuple in UCC_terms[INDEX]]


G = nx.Graph()
G = Build_Graph_Nodes(List_of_nodes, G, node_attributes_dict=None, plot_graph=False)
G = Build_Graph_Edges_COMMUTING_QWC_AntiCommuting(G, List_of_nodes,'C', plot_graph = False)

# comp_G = Get_Complemenary_Graph(G, node_attributes_dict=node_attributes_dict, plot_graph=True) # <- not currently used


single_G, multi_G = Get_subgraphs(G, node_attributes_dict=None)
s_colour = Colour_list_of_Graph(single_G, attribute_dictionary=None, plot_graph=False,
                                strategy='largest_first')
m_colour = Colour_list_of_Graph(multi_G, attribute_dictionary=None, plot_graph=False,
                                strategy='largest_first')

anti_commuting_sets = Get_unique_graph_colours(s_colour + m_colour)