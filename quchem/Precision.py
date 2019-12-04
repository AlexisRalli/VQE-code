from quchem.Hamiltonian_Generator_Functions import Hamiltonian
from quchem.Graph import BuildGraph_string
from quchem.Unitary_partitioning import *
from quchem.Ansatz_Generator_Functions import *
from quchem.Simulating_Quantum_Circuit import *
from tqdm import tqdm
import numpy as np




### Variable Parameters
Molecule = 'H2'
geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.74))]
n_electrons = 2
num_shots = 10000
max_iter = 5
T1_and_T2_theta_list_GUESS = [random.uniform(0, 2*math.pi) for i in range(3)]
####

### Get Hamiltonian
Hamilt = Hamiltonian(Molecule,
                     run_scf = 1, run_mp2 = 1, run_cisd = 0, run_ccsd = 0, run_fci = 1,
                 basis = 'sto-3g',
                 multiplicity = 1,
                 geometry = geometry) # normally None!

Hamilt.Get_all_info(get_FCI_energy=False)


### Ansatz
HF_initial_state= HF_state_generator(n_electrons, Hamilt.MolecularHamiltonian.n_qubits)
#HF_initial_state = [0, 0, 1, 1]
#HF_initial_state = [0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]


HF_UCC = Full_state_prep_circuit(HF_initial_state, T1_and_T2_theta_list=[1.55957373, 1.57789987, 0.78561344])#, T1_and_T2_theta_list=[0,np.pi,0.5*np.pi]) // [1.55957373, 1.57789987, 0.78561344]
HF_UCC.complete_UCC_circuit()
full_anstaz_circuit =HF_UCC.UCC_full_circuit
#print(full_anstaz_circuit)




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

### Get Unitary Partition
zz = UnitaryPartition(anti_commuting_sets, full_anstaz_circuit, S=0)
zz.Get_Quantum_circuits_and_constants()
circuits_and_constants = zz.circuits_and_constants


def No_Measurements_to_obtain_Precision(precision, PauliWord, quantum_circuit, cofactor, max_measure):

    M_i = 1
    epsilon = 1

    while epsilon > precision:

        if max_measure == M_i:
            raise ValueError('Too many measurement required to get desired precision')

        else:
            sim = Simulate_Single_Circuit(PauliWord, quantum_circuit, M_i)
            energy = sim.Get_expectation_value_via_parity()
            epsilon = np.sqrt(cofactor ** 2 * (1 - energy ** 2) / M_i)

            if epsilon == 0:
                epsilon=1
                M_i += 1
                continue
            else:
                epsilon = np.sqrt(cofactor**2 * (1 - energy **2) / M_i)
                M_i += 1
                #print(M_i, epsilon)
    return M_i

i = 10
precision = 0.01
PauliWord = circuits_and_constants[i]['PauliWord']
quantum_circuit = circuits_and_constants[i]['circuit']
cofactor = circuits_and_constants[i]['gamma_l']

max_measure = 10000

#No_Measurements_to_obtain_Precision(precision, PauliWord, quantum_circuit, cofactor, max_measure)

def AVERAGE_No_Measurements_to_obtain_Precision(precision, PauliWord, quantum_circuit, cofactor, max_measure, iter):
    Mi_list =[]

    for i in tqdm(range(iter), ascii=True, desc='Getting average no of measurements for precision'):
        Mi_list.append(No_Measurements_to_obtain_Precision(precision, PauliWord, quantum_circuit,
                                                           cofactor, max_measure))
    return sum(Mi_list)/iter

No_Measure = AVERAGE_No_Measurements_to_obtain_Precision(precision, PauliWord, quantum_circuit, cofactor, max_measure, 20)

sim = Simulate_Single_Circuit(PauliWord, quantum_circuit, round(No_Measure))
sim.Get_expectation_value_via_parity()