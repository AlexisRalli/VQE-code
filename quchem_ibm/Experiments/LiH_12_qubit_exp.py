from quchem_ibm.Qiskit_Chemistry import *
from quchem.Hamiltonian_Generator_Functions import *
from quchem.Graph import *
from openfermion import qubit_operator_sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigs
from scipy.linalg import eig
from quchem.Ansatz_Generator_Functions import *

def main():
    transformation = 'BK'

    ## HAMILTONIAN start

    Molecule = 'LiH'
    geometry = [('Li', (0., 0., 0.)), ('H', (0., 0., 1.45))]
    basis = 'sto-6g'

    ### Get Hamiltonian
    Hamilt = Hamiltonian_PySCF(Molecule,
                               run_scf=1, run_mp2=1, run_cisd=1, run_ccsd=1, run_fci=1,
                               basis=basis,
                               multiplicity=1,
                               geometry=geometry)  # normally None!
    QubitHamiltonian = Hamilt.Get_Qubit_Hamiltonian(threshold=None, transformation=transformation)
    ### HAMILTONIAN end

    #####################################

    print(QubitHamiltonian)

    fci_energy = Hamilt.molecule.fci_energy
    print(fci_energy)


    n_electrons = Hamilt.molecule.n_electrons
    n_qubits = Hamilt.molecule.n_qubits

    ansatz_obj = Ansatz(n_electrons, n_qubits)
    print('JW ground state = ', ansatz_obj.Get_JW_HF_state_in_OCC_basis())
    print('BK ground state = ', ansatz_obj.Get_BK_HF_state_in_OCC_basis())

    NewQubitHamiltonian_relabelled = QubitHamiltonian


    new_Molecular_H_MATRIX = csc_matrix(qubit_operator_sparse(NewQubitHamiltonian_relabelled))

    try:
        eig_values, eig_vectors = eigs(new_Molecular_H_MATRIX)
    except:
        eig_values, eig_vectors = eig(new_Molecular_H_MATRIX.todense())

    new_FCI_Energy = min(eig_values)

    index = np.where(eig_values == new_FCI_Energy)[0][0]
    ground_state_vector = eig_vectors[:, index]

    print('new_FCI = ', new_FCI_Energy, 'VS old FCI:', fci_energy)
    print(np.isclose(new_FCI_Energy, fci_energy))

    new_input_state = ansatz_obj.Get_BK_HF_state_in_OCC_basis() if transformation == 'BK' else ansatz_obj.Get_JW_HF_state_in_OCC_basis()

    ####### ANSATZ
    n_qubits = len(new_input_state)
    check_ansatz_state = False
    Ansatz_circuit, q_reg = Vector_defined_Ansatz(n_qubits, ground_state_vector, check_ansatz_state=check_ansatz_state)

    standard_VQE_circuits, standard_I_term = Build_Standard_VQE_circuits(
        NewQubitHamiltonian_relabelled,
        Ansatz_circuit,
        q_reg)


    Hamiltonian_graph_obj = Openfermion_Hamiltonian_Graph(NewQubitHamiltonian_relabelled)
    commutativity_flag = 'AC'  ## <- defines relationship between sets!!!
    plot_graph = False
    Graph_colouring_strategy = 'largest_first'
    anti_commuting_sets = Hamiltonian_graph_obj.Get_Clique_Cover_as_QubitOp(commutativity_flag,
                                                                            Graph_colouring_strategy=Graph_colouring_strategy,
                                                                            plot_graph=plot_graph)

    print(anti_commuting_sets)

    n_qubits = len(new_input_state)
    check_ansatz_state = False
    rotation_reduction_check = False

    Seq_Rot_VQE_circuits, Seq_Rot_I_term = Get_Seq_Rot_Unitary_Part_circuits(
        anti_commuting_sets,
        Ansatz_circuit,
        q_reg,
        n_qubits,
        S_index_dict=None,
        rotation_reduction_check=rotation_reduction_check)

    n_qubits = len(new_input_state)
    rotation_reduction_check = False

    Seq_Rot_VQE_circuits, Seq_Rot_I_term = Get_Seq_Rot_Unitary_Part_circuits(
        anti_commuting_sets,
        Ansatz_circuit,
        q_reg,
        n_qubits,
        S_index_dict=None,
        rotation_reduction_check=rotation_reduction_check)

    n_qubits = len(new_input_state)
    check_ansatz_state = False

    LCU_VQE_circuits, LCU_I_term = Get_LCU_Unitary_Part_circuits(anti_commuting_sets,
                                                                 Ansatz_circuit,
                                                                 q_reg,
                                                                 n_qubits,
                                                                 N_index_dict=None)
    filename = 'LiH_bravyi_kitaev_12_qubit_experiment'
    n_qubits = len(new_input_state)

    Save_exp_inputs(filename, NewQubitHamiltonian_relabelled, anti_commuting_sets, Hamilt.geometry, basis,
                    transformation,
                    Graph_colouring_strategy, fci_energy,
                    standard_VQE_circuits, standard_I_term,
                    Seq_Rot_VQE_circuits, Seq_Rot_I_term,
                    LCU_VQE_circuits, LCU_I_term,
                    ground_state_vector,
                    n_qubits,
                    S_index_dict=None,
                    N_index_dict=None)


if __name__ == '__main__':
    main()