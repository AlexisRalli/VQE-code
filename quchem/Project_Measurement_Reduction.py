#from .tests.VQE_methods.Hamiltonian_Generator_Functions import
from tests.VQE_methods.Hamiltonian_Generator_Functions import Hamiltonian
from tests.VQE_methods.Graph import BuildGraph_string
from tests.VQE_methods.Unitary_partitioning import *


### Get Hamiltonian
Molecule = 'H2'
Hamilt = Hamiltonian(Molecule,
                     run_scf = 1, run_mp2 = 1, run_cisd = 0, run_ccsd = 0, run_fci = 1,
                 basis = 'sto-3g',
                 multiplicity = 1,
                 geometry = None)

Hamilt.Get_all_info(get_FCI_energy=False)

Commuting_indices = Hamilt.Commuting_indices
PauliWords = Hamilt.QubitHamiltonianCompleteTerms
constants = Hamilt.HamiltonainCofactors

### Build Graph
HamiltGraph = BuildGraph_string(PauliWords, Commuting_indices, constants)
HamiltGraph.Build_string_nodes()  # plot_graph=True)
HamiltGraph.Build_string_edges()  # plot_graph=True)
HamiltGraph.Get_complementary_graph_string()  # plot_graph=True)
HamiltGraph.colouring(plot_graph=False)

anti_commuting_sets = HamiltGraph.anticommuting_sets

### Get Unitary Partition

All_X_sk_terms = X_sk_terms(anti_commuting_sets, S=0)
All_X_sk_terms.Get_all_X_sk_operator()

# print(All_X_sk_terms.normalised_anti_commuting_sets)
# print(All_X_sk_terms.X_sk_Ops)
R_S_operators_by_key = Get_R_S_operators(All_X_sk_terms.X_sk_Ops)
print(cirq.Circuit.from_ops(cirq.decompose_once(
    (R_S_operators_by_key[7][0][0](*cirq.LineQubit.range(R_S_operators_by_key[7][0][0].num_qubits()))))))