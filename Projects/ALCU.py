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

### Ansatz ###
# from quchem.Ansatz_Generator_Functions import *
#
# UCCSD = UCCSD_Trotter(SQ_CC_ops, THETA_params)
# Second_Quant_CC_JW_OP_list = UCCSD.SingleTrotterStep()
# PauliWord_list = Convert_QubitOperator_To_Pauliword_Str_list(Second_Quant_CC_JW_OP_list)
# HF_UCCSD_ansatz = Ansatz_Circuit(PauliWord_list, Hamilt.molecule.n_electrons, Hamilt.molecule.n_qubits)
# # THETA_params = [random.uniform(0, 2 * np.pi) for _ in range(len(THETA_params))]
# ansatz_Q_cicuit = HF_UCCSD_ansatz.Get_Full_HF_UCCSD_QC(THETA_params)
# print(ansatz_Q_cicuit)


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

anti_commuting_set = Get_unique_graph_colours(s_colour + m_colour)
print(anti_commuting_set)

# ALCU!!!
# R = exp(−iαX/2) = cos(α/2)I − isin(α/2)X
# X = i ∑^{n-1}_{k=1} β_{k}P_{k}P_{n}
# therefore:
# R = cos(α/2)I + sin(α/2) ∑^{n-1}_{k=1} β_{k}P_{k} P_{n}
# combining PauliWords (with new constants!)
# R =  ∑_{Ω} β_{Ω}P_{Ω}
## now can perform LCU
# define:
### |A > = 1/||a||_{1}   ∑_{Ω} (β_{Ω})^0.5|Ω >  # ancilla line!
### c-H_{s} = ∑_{Ω} |Ω >< Ω| ⊗ P_{Ω}            # control H_s (controlled by ancilla line)
# applying: