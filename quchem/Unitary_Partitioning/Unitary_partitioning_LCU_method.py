from openfermion.ops import QubitOperator
from openfermion.linalg import qubit_operator_sparse
import numpy as np
from scipy.sparse.linalg import expm


from quchem.Misc_functions.Misc_functions import sparse_allclose
from quchem.Unitary_Partitioning.Unitary_partitioning_Seq_Rot import Normalise_Clique

def Get_R_op_list(anti_commuting_set, N_index, N_Qubits, check_reduction=False, atol=1e-8, rtol=1e-05):#, N_qubits_for_check=None):
    """

    Function gets the R operator as a linear combination of unitary operators.

    First the X operator is found:
    X = i ∑_{k=0} 𝛿_{k} P_{kn}

    R has the definition:
    𝑅=exp(−𝑖𝛼X/2)=cos(𝛼/2)𝟙−𝑖sin(𝛼/2)X
    this is used to build R

    ###
     anti_set = ∑_{i=0} 𝛼_{i} P_{i}.
     normalised = 𝛾_{𝑙} ∑_{i=0} 𝛽_{i} P_{i}... where ∑_{i=0} 𝛽_{i}^{2} =1

     the 𝛽n Pn is then removed and set normalised again:
     H_{n_1} =  Ω_{𝑙} ∑_{k=0} 𝛿_{k} P_{k} ... where k!=n

    then:
    X = i ∑_{k=0} 𝛿_{k} P_{k} P_{n} = i ∑_{k=0} 𝛿_{k} P_{kn}
    ####
    Paper also defines
    H_n = cos(𝜙_{n-1}) Pn + sin(𝜙_{n-1}) H_{n_1 }

    currently have:
    H_{n}/𝛾_{𝑙} = 𝛽n Pn +  Ω_{𝑙} H_{n_1}

    therefore:
    𝜙_{n-1} = arccos(𝛽n)
    as Ω_{𝑙} is always positive, so if 𝜙_{n-1} > 𝜋 ....THEN.... 𝜙_{n-1} = 2𝜋 - arccos(𝛽n)


    Args:
        anti_commuting_set (list): list of anti-commuting qubit operators
        N_index (int): index of term to reduce too
    Returns:
        R_linear_comb_list (list): linear combination of R operators that makes up R operator
        P_n: (QubitOperator): qubit operator to be reduced too (Pn)
        gamma_l (float): normalisation term (𝛾_{𝑙])
     """

    # 𝛾_𝑙 ∑ 𝛽_𝑗 𝑃_𝑗
    normalised_FULL_set = Normalise_Clique(anti_commuting_set)
    gamma_l = normalised_FULL_set['gamma_l']

    norm_FULL_set = normalised_FULL_set['PauliWords'].copy()

    # 𝛽_n 𝑃_n
    qubitOp_Pn_beta_n = norm_FULL_set.pop(N_index)

    # Ω_𝑙 ∑ 𝛿_k 𝑃_k  ... note this doesn't contain 𝛽_n 𝑃_n
    H_n_1 = Normalise_Clique(norm_FULL_set)
    Omega_l = H_n_1['gamma_l']

    ##

    # cos(𝜙_{𝑛−1}) =𝛽_𝑛
    phi_n_1 = np.arccos(list(qubitOp_Pn_beta_n.terms.values())[0])

    # require sin(𝜙_{𝑛−1}) to be positive...
    # this uses CAST diagram to ensure the sign term is positive and cos term has correct sign (can be negative)
    if (phi_n_1 > np.pi):
        # ^ as sin phi_n_1 must be positive phi_n_1 CANNOT be larger than 180 degrees!
        phi_n_1 = 2 * np.pi - phi_n_1
        print('correct quadrant found!!!')

    #     𝑅=exp(−𝑖𝛼 X/2)=cos(𝛼/2)𝟙 − 𝑖sin(𝛼/2)X = cos(𝛼/2)𝟙 − 𝑖sin(𝛼/2)(i∑𝛿𝑘 𝑃𝑘𝑃𝑛)
    #     𝑅=exp(−𝑖𝛼 X/2)=cos(𝛼/2)𝟙 − 𝑖sin(𝛼/2)X = cos(𝛼/2)𝟙 + sin(𝛼/2)(∑𝛿𝑘 𝑃𝑘𝑃𝑛) #<--- note sign here!
    Pn = QubitOperator(list(qubitOp_Pn_beta_n.terms.keys())[0],
                       1)  # np.sign(list(qubitOp_Pn_beta_n.terms.values())[0]))

    alpha = phi_n_1.copy()
    #     print('alpha/2 =', (alpha/(2*np.pi))*360/2)

    I_term = QubitOperator('', np.cos(alpha / 2))
    R_linear_comb_list = [I_term]

    sin_term = -np.sin(alpha / 2)

    for qubitOp_Pk in H_n_1['PauliWords']:
        PkPn = qubitOp_Pk * Pn
        R_linear_comb_list.append(sin_term * PkPn)

    if not np.isclose(sum(np.absolute(list(qubitOp.terms.values())[0]) ** 2 for qubitOp in R_linear_comb_list), 1):
        raise ValueError(
            'normalisation of X operator incorrect: {}'.format(sum(np.absolute(list(qubitOp.terms.values())[0]) ** 2
                                                                   for qubitOp in R_linear_comb_list)))

    if check_reduction:
        R = QubitOperator()
        for op in R_linear_comb_list:
            R += op
        
        Pn_mat = qubit_operator_sparse(Pn, n_qubits=N_Qubits)
        R_mat = qubit_operator_sparse(R, n_qubits=N_Qubits)

        H_S = QubitOperator()
        for QubitOp in normalised_FULL_set['PauliWords']:
            H_S += QubitOp
        H_S_matrix = qubit_operator_sparse(H_S, n_qubits=N_Qubits)

        RHR_dag = R_mat.dot(H_S_matrix.dot(R_mat.conj().transpose())) 
        if not  sparse_allclose(Pn_mat, RHR_dag, atol=atol, rtol=rtol): # checking R.H_{l}.R† == Pn
            raise ValueError('error in unitary partitioning reduction: R H_s R† != Pn')

    return R_linear_comb_list, Pn, gamma_l  


### LCU operator new check method ###

def LCU_Check(AC_set, N_index, N_Qubits, atol=1e-8, rtol=1e-05):
    if len(AC_set) < 2:
        raise ValueError('no unitary partitioning possible for set sizes less than 2')
    R_uncorrected, Pn, gamma_l = Get_R_op_list(AC_set, N_index)
    #     R_corrected_Op_list, R_corr_list, ancilla_amplitudes, l1_norm = absorb_complex_phases(R_uncorrected)

    R = QubitOperator()
    for op in R_uncorrected:
        R += op

    Pn_mat = qubit_operator_sparse(Pn, n_qubits=N_Qubits)
    R_mat = qubit_operator_sparse(R, n_qubits=N_Qubits)

    full_normalised_set = Normalise_Clique(AC_set)

    H_S = QubitOperator()
    for QubitOp in full_normalised_set['PauliWords']:
        H_S += QubitOp
    H_S_matrix = qubit_operator_sparse(H_S, n_qubits=N_Qubits)

    RHR_dag = R_mat.dot(H_S_matrix.dot(R_mat.conj().transpose())) 
    return sparse_allclose(Pn_mat, RHR_dag, atol=atol, rtol=rtol) # R.H_{l}.R† == Pn

from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from scipy.sparse import csc_matrix
def LCU_linalg_Energy(anti_commuting_sets, N_indices_dict, N_Qubits, atol=1e-8, rtol=1e-05, check_reduction=False):
    reduced_H_matrix = csc_matrix((2 ** N_Qubits, 2 ** N_Qubits), dtype=complex)

    H_single_terms = QubitOperator()

    for key in anti_commuting_sets:
        AC_set = anti_commuting_sets[key]

        if len(AC_set) < 2:
            H_single_terms += AC_set[0]
        else:
            N_index = N_indices_dict[key]

            R_uncorrected, Pn, gamma_l = Get_R_op_list(AC_set, N_index, N_Qubits, check_reduction=check_reduction, atol=atol, rtol=rtol)
            #     R_corrected_Op_list, R_corr_list, ancilla_amplitudes, l1_norm = absorb_complex_phases(R_uncorrected)

            R = QubitOperator()
            for op in R_uncorrected:
                R += op

            R_mat = qubit_operator_sparse(R, n_qubits=N_Qubits)
            Pn_mat = qubit_operator_sparse(Pn, n_qubits=N_Qubits)

            RPR_matrix = R_mat.conj().transpose().dot(
                Pn_mat.dot(R_mat))  # note this is R^{dag}PR and NOT: RHR^{dag}

            if check_reduction:
                full_normalised_set = Normalise_Clique(AC_set)
                H_S = QubitOperator()
                for QubitOp in full_normalised_set['PauliWords']:
                    H_S += QubitOp
                H_S_matrix = qubit_operator_sparse(H_S, n_qubits=N_Qubits)

                RHR_matrix = R_mat.dot(H_S_matrix.dot(R_mat.conj().transpose()))
                if sparse_allclose(Pn_mat, RHR_matrix, atol=atol, rtol=rtol) is not True:
                    raise ValueError('error in unitary partitioning reduction')

            reduced_H_matrix += RPR_matrix * gamma_l

    reduced_H_matrix += qubit_operator_sparse(H_single_terms, n_qubits=N_Qubits)
    # eig_values, eig_vectors = sparse_eigs(reduced_H_matrix)
    if N_Qubits<4:
        eig_values, eig_vectors = eigh(reduced_H_matrix.todense()) # NOT sparse!
    else:
        eig_values, eig_vectors = eigsh(reduced_H_matrix, k=1, which='SA') # < solves eigenvalue problem for a complex Hermitian matrix.
    FCI_Energy = min(eig_values)
    return FCI_Energy