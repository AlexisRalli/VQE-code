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

### Graph Colouring
from quchem.Graph import *

List_PauliWords, HamiltonainCofactors = zip(*QubitHam_PauliStr)

attribute_dictionary = {'Cofactors': HamiltonainCofactors}

List_of_nodes, node_attributes_dict = Get_list_of_nodes_and_attributes(List_PauliWords,
                                                                       attribute_dictionary=attribute_dictionary)

G =  Hamiltonian_Graph(List_PauliWords, Graph_colouring_strategy='largest_first', attribute_dictionary=attribute_dictionary)
anti_commuting_sets = G.Get_Pauli_grouping('C', plot_graph=False)
print(anti_commuting_sets)

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

# note: https://arxiv.org/pdf/quant-ph/0104030.pdf

from scipy.sparse import kron
from functools import reduce
zero = np.array([[1],
                 [0]])

one = np.array([[0],
                [1]])

tensored_PauliWord = reduce(kron, [zero,zero,zero])