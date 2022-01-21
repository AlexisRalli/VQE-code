import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix
import itertools
from copy import deepcopy
from openfermion import hermitian_conjugated
from openfermion.ops import QubitOperator
import quchem.Unitary_Partitioning.Unitary_partitioning_LCU_method as UP_LCU

import cs_vqe as c
from functools import reduce
import quchem.Misc_functions.conversion_scripts as conv_scr 
from quchem.Unitary_Partitioning.Unitary_partitioning_Seq_Rot import Get_Xsk_op_list

def diagonalize_epistemic_SeqRot(model,fn_form,ep_state, check_reduction=False):
    
    assert(len(ep_state[0]) == fn_form[0])
    assert(len(model[0]) == fn_form[0])
    assert(len(ep_state[1]) == fn_form[1])
    assert(len(model[1]) == fn_form[1])

    R_sk_OP_list = None
    # if there are cliques...
    if fn_form[1] > 0:
        # rotations to map A to a single Pauli (to be applied on left)

        script_A = [conv_scr.convert_op_str(op_A, coeff) for op_A, coeff in zip(model[1], ep_state[1])] #AC set (note has already been normalized, look at eq 17 of contextual VQE paper!)
        
        S_index=0
        N_Qubits = len(model[1][0])
        X_sk_theta_sk_list, normalised_FULL_set, Ps, gamma_l = Get_Xsk_op_list(script_A,
                                                                                S_index,
                                                                                N_Qubits,
                                                                                check_reduction=check_reduction,
                                                                                atol=1e-8,
                                                                                rtol=1e-05)
        GuA = deepcopy(model[0] + [model[1][0]]) #<--- here N_index fixed to zero (model[1][0] == first op of script_A !) (TODO: could be potential bug if other vals used)
        ep_state_trans = deepcopy(ep_state[0] + [1]) # < - fixes script A eigenvalue!
        
        R_sk_OP_list = []
        for X_sk_Op, theta_sk in X_sk_theta_sk_list:
            op = np.cos(theta_sk / 2) * QubitOperator('') -1j*np.sin(theta_sk / 2) * X_sk_Op
            R_sk_OP_list.append(op)
        # R_S_op = reduce(lambda x,y: x*y, R_sk_OP_list[::-1])  # <- note reverse order!
        # R_SeqRot = list(R_S_op)
        # print(gamma_l)

    # if there are no cliques...
    else:
        # rotations to diagonalize G
        GuA = deepcopy(model[0])
        ep_state_trans = deepcopy(ep_state[0])

    rotations_dict = {}
    for i in range(len(GuA)):
        rotations = []
        g = GuA[i]

        # if g is not already a single Z...
        if not any((all(g[k] == 'I' or k == j for k in range(len(g))) and g[j] == 'Z') for j in range(len(g))):
        
            # if g is diagonal...
            if all(p == 'I' or p == 'Z' for p in g):
                
                # store locations where g has a Z and none of the previously diagonalized generators do
                Zs = []
                for m in range(len(g)):
                    if g[m] == 'Z' and all(h[m] == 'I' for h in GuA[:i]):
                        Zs.append(m)
                        
                # there must be at least one such location: pick the first one
                assert(len(Zs) > 0)
                m = Zs[0]
                
                # construct a rotation about the single Y operator acting on qubit m
                K = ''
                for o in range(len(g)):
                    if o == m:
                        K += 'Y'
                    else:
                        K += 'I'
                
                # add adjoint rotation to rotations list
                rotations.append(['pi/2', K])
                
                # apply R to GuA
                for m in range(len(GuA)):
                    if not c.commute(GuA[m],K):
                        p = deepcopy(c.pauli_mult(K,GuA[m]))
                        GuA[m] = p[0]
                        ep_state_trans[m] = 1j*p[1]*ep_state_trans[m]
        
            g = GuA[i]
            # g should now not be diagonal
            if not any(p != 'I' and p != 'Z' for p in g):
                print(model,'\n')
                print(fn_form,'\n')
                print(ep_state,'\n')
                print(GuA)
                print(g)
            assert(any(p != 'I' and p != 'Z' for p in g))
        
            # construct a rotation to map g to a single Z
            J = ''
            found = False
            for j in range(len(g)):
                if g[j] == 'X':
                    if found:
                        J += 'X'
                    else:
                        J += 'Y'
                        found = True
                elif g[j] == 'Y':
                    if found:
                        J += 'Y'
                    else:
                        J += 'X'
                        found = True
                else:
                    J += g[j]
        
            # add adjoint rotation to rotations list
            rotations.append( ['pi/2', J] )
        
            # apply R to GuA
            for m in range(len(GuA)):
                if not c.commute(GuA[m],J):
                    p = deepcopy(c.pauli_mult(J,GuA[m]))
                    GuA[m] = p[0]
                    ep_state_trans[m] = 1j*p[1]*ep_state_trans[m]

        rotations_dict[g] = {'rotations':rotations, 'ep_state_trans': np.real(ep_state_trans)}

    return R_sk_OP_list, rotations_dict, GuA


# Given a Hamiltonian ham, for which the noncontextual part has a quasi-quantized model
# specified by model and fn_form, and a noncontextual ground state specified by ep_state,
# returns the noncontextual ground state energy plus the quantum correction.
def quantum_correction_SeqRot(ham,model,fn_form,ep_state, check_reduction=False):
    
    R_SeqRot, rotations, diagonal_set, vals = diagonalize_epistemic_SeqRot(model,
                                                                           fn_form,
                                                                           ep_state,
                                                                           check_reduction=check_reduction)

#     print(diagonal_set)
#     print(vals)
#     print(rotations)
    
    n_q = len(diagonal_set[0])
    
    ham_rotated = deepcopy(ham)
    
    for r in rotations: # rotate the full Hamiltonian to the basis with diagonal noncontextual generators
        ham_next = {}
        for t in ham_rotated.keys():
            t_set_next = c.apply_rotation(r,t)
            for t_next in t_set_next.keys():
                if t_next in ham_next.keys():
                    ham_next[t_next] = ham_next[t_next] + t_set_next[t_next]*ham_rotated[t]
                else:
                    ham_next[t_next] = t_set_next[t_next]*ham_rotated[t]
        ham_rotated = deepcopy(ham_next)
        
#     print(ham_rotated)
       
    z_indices = []
    for d in diagonal_set:
        for i in range(n_q):
            if d[i] == 'Z':
                z_indices.append(i)
                
#     print(z_indices)
        
    ham_red = {}
    
    for t in ham_rotated.keys():
        
        sgn = 1
        
        for j in range(len(diagonal_set)): # enforce diagonal generator's assigned values in diagonal basis
            z_index = z_indices[j]
            if t[z_index] == 'Z':
                sgn = sgn*vals[j]
            elif t[z_index] != 'I':
                sgn = 0
        
        if sgn != 0:
            # construct term in reduced Hilbert space
            t_red = ''
            for i in range(n_q):
                if not i in z_indices:
                    t_red = t_red + t[i]
            if t_red in ham_red.keys():
                ham_red[t_red] = ham_red[t_red] + ham_rotated[t]*sgn
            else:
                ham_red[t_red] = ham_rotated[t]*sgn
        
#         print(t,t_red,sgn,ham_rotated[t])
            
#     print('\n\n',ham_red)
    
    if n_q-len(diagonal_set) == 0:
        assert(len(list(ham_red.keys())) == 1)
        assert(list(ham_red.keys())[0] == '')
        return list(ham_red.values())[0].real
    
    else:
        # find lowest eigenvalue of reduced Hamiltonian
        ham_red_sparse = c.hamiltonian_to_sparse(ham_red)
        if n_q-len(diagonal_set) <= 4:
            return min(np.linalg.eigvalsh(ham_red_sparse.toarray()))
        else:
            return sp.sparse.linalg.eigsh(ham_red_sparse, which='SA', k=1)[0][0]


# Given ham (the full Hamiltonian), model (the quasi-quantized model for the noncontextual part),
# fn_form (the output of energy_function_form(ham_noncon,model)), and order (a list specifying the order in which to remove the qubits),
# returns a list whose elements are the reduced quantum search Hamiltonians for contextual subspace VQE,
# where the ith element corresponds to simulating i qubits on the quantum computer,
# with the remaining qubits simulated by the noncontextual approximation.
# The noncontextual ground state energy is included in the constant term of each Hamiltonian,
# so the complete CS-VQE approximation is obtained by finding the ground state energy of each.
# If order is shorter than the total number of qubits, only Hamiltonians up to the the number of qubits
# reflected by order are returned.
def get_reduced_hamiltonians_SeqRot(ham,model,fn_form,ep_state,order, check_reduction=False):

    R_SeqRot, rotations, diagonal_set, vals = diagonalize_epistemic_SeqRot(model,fn_form,ep_state, check_reduction=check_reduction)
    
    n_q = len(diagonal_set[0])
    
    order_len = len(order)
    
    vals = list(vals)
    
    # rectify order
    for i in range(len(order)):
        for j in range(i):
            if order[j] < order[i]:
                order[i] -= 1
    
    out = []


    if R_SeqRot is not None:
        rot_H = conv_scr.Get_Openfermion_Hamiltonian(ham)
        for rot in R_SeqRot:
            H_next = QubitOperator()
            for t in rot_H:
                t_set_next = rot * t * hermitian_conjugated(rot)
                H_next+=t_set_next
            rot_H = deepcopy(list(H_next))

        post_SeqRot_rot_ham = conv_scr.Openfermion_to_dict(rot_H, n_q)
        post_SeqRot_rot_ham_real_and_pruned = {P_key: coeff.real for P_key, coeff in post_SeqRot_rot_ham.items() if not np.isclose(coeff.real,0)}
        del rot_H
        del H_next
    else:
        post_SeqRot_rot_ham_real_and_pruned = {P_key: coeff.real for P_key, coeff in ham.items() if not np.isclose(coeff.real,0)}


    for k in range(order_len+1):
        
        ham_rotated = deepcopy(post_SeqRot_rot_ham_real_and_pruned)

        for r in rotations: # rotate the full Hamiltonian to the basis with diagonal noncontextual generators
            ham_next = {}
            for t in ham_rotated.keys():
                t_set_next = c.apply_rotation(r,t)
                for t_next in t_set_next.keys():
                    if t_next in ham_next.keys():
                        ham_next[t_next] = ham_next[t_next] + t_set_next[t_next]*ham_rotated[t]
                    else:
                        ham_next[t_next] = t_set_next[t_next]*ham_rotated[t]
            ham_rotated = deepcopy(ham_next)
       
        z_indices = []
        for d in diagonal_set:
            for i in range(n_q):
                if d[i] == 'Z':
                    z_indices.append(i)
        
        ham_red = {}
    
        for t in ham_rotated.keys():
        
            sgn = 1
        
            for j in range(len(diagonal_set)): # enforce diagonal generator's assigned values in diagonal basis
                z_index = z_indices[j]
                if t[z_index] == 'Z':
                    sgn = sgn*vals[j]
                elif t[z_index] != 'I':
                    sgn = 0
        
            if sgn != 0:
                # construct term in reduced Hilbert space
                t_red = ''
                for i in range(n_q):
                    if not i in z_indices:
                        t_red = t_red + t[i]
                if t_red in ham_red.keys():
                    ham_red[t_red] = ham_red[t_red] + ham_rotated[t]*sgn
                else:
                    ham_red[t_red] = ham_rotated[t]*sgn

        out.append(ham_red)

        if order:
            # Drop a qubit:
            i = order[0]
            order.remove(i)
            diagonal_set = diagonal_set[:i]+diagonal_set[i+1:]
            vals = vals[:i]+vals[i+1:]
    
    return out


# Given ham (the full Hamiltonian), model (the quasi-quantized model for the noncontextual part),
# fn_form (the output of energy_function_form(ham_noncon,model)), and order (a list specifying the order in which to remove the qubits),
# returns a list of approximations to the ground state energy of the full Hamiltonian,
# whose ith element is the approximation obtained by simulating i qubits on the quantum computer,
# with the remaining qubits simulated by the noncontextual approximation.
# (Hence the 0th element is the pure noncontextual approximation.)
# If order is shorter than the total number of qubits, only approximations using qubit removals
# in order are simulated.
def contextual_subspace_approximations_SeqRot(ham,model,fn_form,ep_state, order, check_reduction=False):
    

    reduced_hamiltonians = get_reduced_hamiltonians_SeqRot(ham,model,fn_form,ep_state,order, check_reduction=check_reduction)

    n_q = len(list(ham.keys())[0])

    out = []

    for ham_red in reduced_hamiltonians:
    
        if len(list(ham_red.keys())) == 1:
            assert(list(ham_red.keys())[0] == '')
            out.append(list(ham_red.values())[0].real)
    
        else:
            # find lowest eigenvalue of reduced Hamiltonian
            ham_red_sparse = c.hamiltonian_to_sparse(ham_red)
            if ham_red_sparse.shape[0] <= 64: #len(list(ham_red.keys())) <= 6:
                out.append(min(np.linalg.eigvalsh(ham_red_sparse.toarray())))
            else:
#                 print(f'  computing restricted ground state for {n_q-len(diagonal_set)} qubits...')
                out.append(sp.sparse.linalg.eigsh(ham_red_sparse, which='SA', k=1)[0][0])

    return out

# Heuristic search for best qubit removal ordering:
# search starting from all qubits in noncontextual part,
# and moving them to qc two at a time,
# greedily choosing the pair that maximally reduces overall error at each step.
def csvqe_approximations_heuristic_SeqRot(ham, ham_noncon, n_qubits, true_gs, check_reduction=False):

    model = c.quasi_model(ham_noncon)
    fn_form = c.energy_function_form(ham_noncon,model)
        
    gs_noncon = c.find_gs_noncon(ham_noncon,method = 'differential_evolution')
        
    if gs_noncon[0]-true_gs < 10**-10:
        return [true_gs,[gs_noncon[0] for i in range(n_qubits)],[gs_noncon[0]-true_gs for i in range(n_qubits)],[i for i in range(n_qubits)]]
            
    else:
            
        ep_state = gs_noncon[1]
        
        indices = [j for j in range(n_qubits)]
        order = []
        
        exact = False
        
        while len(indices)>1 and not exact:
            # print('indices',indices,'\n')
            # print('order',order,'\n')
            num_to_remove = 2
            i_subset_improvements = {i_subset:0 for i_subset in itertools.combinations(indices,num_to_remove)}
            for i_subset in itertools.combinations(indices,num_to_remove):
                possible_order = order+list(i_subset)
                current_i = len(possible_order)
                approxs_temp = contextual_subspace_approximations_SeqRot(ham,model,fn_form,ep_state, possible_order, check_reduction=check_reduction)
                errors_temp = [a - true_gs for a in approxs_temp]
                if errors_temp[current_i] < 10**-6:
                    exact = True
                improvement = errors_temp[current_i-2]-errors_temp[current_i]
                # print([round(e,3) for e in errors_temp],improvement)
                i_subset_improvements[i_subset] += improvement
                    
            best_i_subset = max(i_subset_improvements, key=i_subset_improvements.get)
            # print('\nbest_i_subset',best_i_subset,'\n')
            order = order+list(best_i_subset)
            # print(f'current order: {order}\n')
            indices = []
            for i in range(n_qubits):
                if not i in order:
                    indices.append(i)
                        
        # add last index if necessary
        order_full = deepcopy(order)
        for i in range(n_qubits):
            if not i in order_full:
                order_full.append(i)
                    
        order2 = []
        for i in range(int(len(order)/2)):
            order2.append(order[2*i+1])
            order2.append(order[2*i])
                
        # print(order)
        # print(order2,'\n')
            
        # print('  getting final approximations...\n')

        approxs = contextual_subspace_approximations_SeqRot(ham,model,fn_form,ep_state,deepcopy(order_full), check_reduction=check_reduction)
        errors = [a - true_gs for a in approxs]
            
        # print(errors,'\n')
            
        approxs2 = contextual_subspace_approximations_SeqRot(ham,model,fn_form,ep_state,deepcopy(order2), check_reduction=check_reduction)
        errors2 = [a - true_gs for a in approxs2]
            
        # print(errors2,'\n')
            
        order_out = []
        approxs_out = [approxs[0]]
        errors_out = [errors[0]]
            
        for i in range(int(len(order)/2)):
            if errors[2*i+1] <= errors2[2*i+1]:
                order_out.append(order[2*i])
                order_out.append(order[2*i+1])
                approxs_out.append(approxs[2*i+1])
                approxs_out.append(approxs[2*i+2])
                errors_out.append(errors[2*i+1])
                errors_out.append(errors[2*i+2])
            else:
                order_out.append(order2[2*i])
                order_out.append(order2[2*i+1])
                approxs_out.append(approxs2[2*i+1])
                approxs_out.append(approxs2[2*i+2])
                errors_out.append(errors2[2*i+1])
                errors_out.append(errors2[2*i+2])
                    
        for i in range(len(order),len(order_full)):
            order_out.append(order_full[i])
            approxs_out.append(approxs[i+1])
            errors_out.append(errors[i+1])

        # print('FINAL ORDER',order_out,'\n')
        # print('FINAL ERRORS',errors_out,'\n')
        
        return [true_gs, approxs_out, errors_out, order_out]


# USEFUL code below


def diagonalize_epistemic_dictionary_generator(model, fn_form, ep_state, check_reduction=False, up_method='SeqRot'):
    """
    Function to find mapping of GuA to single qubit Pauli Z
    This can be used to determine which stabilizers Will fixes.

    Args:
        model:
        fn_form:

    Returns:

    """
    assert (len(ep_state[0]) == fn_form[0])
    assert (len(model[0]) == fn_form[0])
    assert (len(ep_state[1]) == fn_form[1])
    assert (len(model[1]) == fn_form[1])

    # if there are cliques...
    R_unitary_part_dict = None
    if fn_form[1] > 0:

        # AC set (note has already been normalized, look at eq 17 of contextual VQE paper!)
        script_A = [conv_scr.convert_op_str(op_A, coeff) for op_A, coeff in zip(model[1], ep_state[1])]

        if up_method == 'SeqRot':
            S_index = 0
            N_Qubits = len(model[1][0])
            X_sk_theta_sk_list, normalised_FULL_set, Ps, gamma_l = Get_Xsk_op_list(script_A,
                                                                                   S_index,
                                                                                   N_Qubits,
                                                                                   check_reduction=check_reduction,
                                                                                   atol=1e-8,
                                                                                   rtol=1e-05)
            R_upart_op_list = []
            for X_sk_Op, theta_sk in X_sk_theta_sk_list:
                op = np.cos(theta_sk / 2) * QubitOperator('') - 1j * np.sin(theta_sk / 2) * X_sk_Op
                R_upart_op_list.append(op)
        elif up_method == 'LCU':
            N_index = 0
            N_Qubits = len(model[1][0])
            R_LCU_list, Pn, gamma_l = UP_LCU.Get_R_op_list(script_A, N_index, N_Qubits, check_reduction=check_reduction,
                                                           atol=1e-8, rtol=1e-05)
            R_upart_op_list = reduce(lambda x, y: x + y, R_LCU_list)
        else:
            raise ValueError(f'unknown unitary partitioning method: {up_method}')

        GuA = deepcopy(model[0] + [model[1][
                                       0]])  # <--- here N_index fixed to zero (model[1][0] == first op of script_A !) (TODO: could be potential bug if other vals used)
        ep_state_trans = deepcopy(ep_state[0] + [1])  # < - fixes script A eigenvalue!
        R_unitary_part_dict = {'unitary_part_method': up_method, 'R_op': R_upart_op_list}
    else:
        # rotations to diagonalize G
        GuA = deepcopy(model[0])
        ep_state_trans = deepcopy(ep_state[0])

    GuA_fixed = deepcopy(GuA)
    ep_state_fixed = deepcopy(ep_state_trans)
    mapping_GuA_to_singleZ_with_ep_exp_vals = {}
    for i in range(len(GuA)):
        rotations = []
        g = GuA[i]

        # if g is not already a single Z...
        if not any((all(g[k] == 'I' or k == j for k in range(len(g))) and g[j] == 'Z') for j in range(len(g))):

            # if g is diagonal...
            if all(p == 'I' or p == 'Z' for p in g):

                # store locations where g has a Z and none of the previously diagonalized generators do
                Zs = []
                for m in range(len(g)):
                    if g[m] == 'Z' and all(h[m] == 'I' for h in GuA[:i]):
                        Zs.append(m)

                # there must be at least one such location: pick the first one
                assert (len(Zs) > 0)
                m = Zs[0]

                # construct a rotation about the single Y operator acting on qubit m
                K = ''
                for o in range(len(g)):
                    if o == m:
                        K += 'Y'
                    else:
                        K += 'I'

                # add adjoint rotation to rotations list
                rotations.append(['pi/2', K])

                # apply R to GuA
                for m in range(len(GuA)):
                    if not c.commute(GuA[m], K):
                        p = deepcopy(c.pauli_mult(K, GuA[m]))
                        GuA[m] = p[0]

            g = GuA[i]
            # g should now not be diagonal
            if not any(p != 'I' and p != 'Z' for p in g):
                print(model, '\n')
                print(fn_form, '\n')
                print(GuA)
                print(g)
            assert (any(p != 'I' and p != 'Z' for p in g))

            # construct a rotation to map g to a single Z
            J = ''
            found = False
            for j in range(len(g)):
                if g[j] == 'X':
                    if found:
                        J += 'X'
                    else:
                        J += 'Y'
                        found = True
                elif g[j] == 'Y':
                    if found:
                        J += 'Y'
                    else:
                        J += 'X'
                        found = True
                else:
                    J += g[j]

            # add adjoint rotation to rotations list
            rotations.append(['pi/2', J])

            # apply R to GuA
            for m in range(len(GuA)):
                if not c.commute(GuA[m], J):
                    p = deepcopy(c.pauli_mult(J, GuA[m]))
                    GuA[m] = p[0]


        if (R_unitary_part_dict is not None) and (i==len(GuA)-1):
            mapping_GuA_to_singleZ_with_ep_exp_vals[GuA_fixed[i]] = {'single_Z': GuA[i], 'noncon_gs_exp_val': ep_state_fixed[i],
                                                    'do_unitary_part': True}
        else:
            mapping_GuA_to_singleZ_with_ep_exp_vals[GuA_fixed[i]] = {'single_Z': GuA[i], 'noncon_gs_exp_val': ep_state_fixed[i],
                                                    'do_unitary_part': False}

    # need to find out Will's ordering of qubit removals so need all generators mapped to single Z!
    GuA_full_rotated_set = GuA
    return mapping_GuA_to_singleZ_with_ep_exp_vals, R_unitary_part_dict, GuA_full_rotated_set


def diagonalize_epistemic_by_fixed_qubit(model, conversion_dict, qubits_to_fix):
    """
    Use conversion dict (contains original GuA and their single qubit translated form) and from qubits to fix map back
    from single qubit Z to original GuA... THEN find rotations to give single qubit Z (note rotations depend on previous
    rotations, hence why this is a very roundabout way of doing this. TODO: specifiy generator fixed only!)

    Args:
        model:
        conversion_dict:
        qubits_to_fix:

    Returns:

    """
    GuA = []
    ep_state_trans=[]
    do_unitary_part = False
    for pauli in conversion_dict.keys():
        Z_op= conversion_dict[pauli]['single_Z']
        Z_ind = Z_op.index('Z')
        if Z_ind in qubits_to_fix:
            GuA.append(pauli)
            ep_state_trans.append(conversion_dict[pauli]['noncon_gs_exp_val'])

            if conversion_dict[pauli]['do_unitary_part']:
                do_unitary_part = True

    rotations = []
    for i in range(len(GuA)):
        g = GuA[i]

        # if g is not already a single Z...
        if not any((all(g[k] == 'I' or k == j for k in range(len(g))) and g[j] == 'Z') for j in range(len(g))):

            # if g is diagonal...
            if all(p == 'I' or p == 'Z' for p in g):

                # store locations where g has a Z and none of the previously diagonalized generators do
                Zs = []
                for m in range(len(g)):
                    if g[m] == 'Z' and all(h[m] == 'I' for h in GuA[:i]):
                        Zs.append(m)

                # there must be at least one such location: pick the first one
                assert (len(Zs) > 0)
                m = Zs[0]

                # construct a rotation about the single Y operator acting on qubit m
                K = ''
                for o in range(len(g)):
                    if o == m:
                        K += 'Y'
                    else:
                        K += 'I'

                # add adjoint rotation to rotations list
                rotations.append(['pi/2', K])

                # apply R to GuA
                for m in range(len(GuA)):
                    if not c.commute(GuA[m], K):
                        p = deepcopy(c.pauli_mult(K, GuA[m]))
                        GuA[m] = p[0]
                        ep_state_trans[m] = 1j * p[1] * ep_state_trans[m]

            g = GuA[i]
            # g should now not be diagonal
            if not any(p != 'I' and p != 'Z' for p in g):
                print(model, '\n')
                print(GuA)
                print(g)
            assert (any(p != 'I' and p != 'Z' for p in g))

            # construct a rotation to map g to a single Z
            J = ''
            found = False
            for j in range(len(g)):
                if g[j] == 'X':
                    if found:
                        J += 'X'
                    else:
                        J += 'Y'
                        found = True
                elif g[j] == 'Y':
                    if found:
                        J += 'Y'
                    else:
                        J += 'X'
                        found = True
                else:
                    J += g[j]

            # add adjoint rotation to rotations list
            rotations.append(['pi/2', J])

            # apply R to GuA
            for m in range(len(GuA)):
                if not c.commute(GuA[m], J):
                    p = deepcopy(c.pauli_mult(J, GuA[m]))
                    GuA[m] = p[0]
                    ep_state_trans[m] = 1j * p[1] * ep_state_trans[m]

    return rotations, do_unitary_part, ep_state_trans, GuA


def get_reduced_hamiltonian_by_qubits_fixed(ham, model, conversion_dict, qubits_to_fix_ordered, R_unitary_part_dict):

    rotations, do_unitary_part, vals, diagonal_set = diagonalize_epistemic_by_fixed_qubit(model,
                                                                                          conversion_dict,
                                                                                          qubits_to_fix_ordered)

    n_q = len(diagonal_set[0])
    vals = list(vals)

    if do_unitary_part is True:

        # Sequence of rotations!
        if R_unitary_part_dict['unitary_part_method'] == 'SeqRot':
            R_unitary_part = R_unitary_part_dict['R_op']

            # do unitary part rotation!
            rot_H = conv_scr.Get_Openfermion_Hamiltonian(ham)
            for rot in R_unitary_part:
                H_next = QubitOperator()
                for t in rot_H:
                    t_set_next = rot * t * hermitian_conjugated(rot)
                    H_next+=t_set_next
                rot_H = deepcopy(list(H_next))

            post_up_ham = conv_scr.Openfermion_to_dict(rot_H, n_q)
            del H_next
            
        elif R_unitary_part_dict['unitary_part_method'] == 'LCU':
            Ham_openF = conv_scr.Get_Openfermion_Hamiltonian(ham)
            R_LCU = R_unitary_part_dict['R_op']
            rot_H = R_LCU * Ham_openF * hermitian_conjugated(R_LCU)  # R H R_dagger!

            post_up_ham = conv_scr.Openfermion_to_dict(rot_H, n_q)
            del R_LCU
        else:
            err = R_unitary_part_dict['unitary_part_method']
            raise ValueError(f'unknown unitary part method: {err}')

        #prune small coeffs!
        ham_rotated = {P_key: coeff.real for P_key, coeff in post_up_ham.items() if not np.isclose(coeff.real,0)}
        del rot_H
        del post_up_ham
    else:
        ham_rotated = deepcopy(ham)

    # rotate stabilizers
    for r in rotations:  # rotate the full Hamiltonian to the basis with diagonal noncontextual generators
        ham_next = {}
        for t in ham_rotated.keys():
            t_set_next = c.apply_rotation(r, t)
            for t_next in t_set_next.keys():
                if t_next in ham_next.keys():
                    ham_next[t_next] = ham_next[t_next] + t_set_next[t_next] * ham_rotated[t]
                else:
                    ham_next[t_next] = t_set_next[t_next] * ham_rotated[t]
        ham_rotated = deepcopy(ham_next)

    z_indices = []
    for d in diagonal_set:
        for i in range(n_q):
            if d[i] == 'Z':
                z_indices.append(i)

    # print('z ind:', z_indices)
    # print('order:', qubits_to_fix_ordered)
    # print(sorted(diagonal_set, key=lambda x: x.index('Z')))
    # print()

    ham_red = {}
    for t in ham_rotated.keys():
        sgn = 1
        for j in range(len(diagonal_set)):  # enforce diagonal generator's assigned values in diagonal basis
            z_index = z_indices[j]
            if t[z_index] == 'Z':
                sgn = sgn * vals[j]
            elif t[z_index] != 'I':
                sgn = 0
        if sgn != 0:
            # construct term in reduced Hilbert space
            t_red = ''
            for i in range(n_q):
                if not i in z_indices:
                    t_red = t_red + t[i]
            if t_red in ham_red.keys():
                ham_red[t_red] = ham_red[t_red] + ham_rotated[t] * sgn
            else:
                ham_red[t_red] = ham_rotated[t] * sgn

    return ham_red


def get_reduced_hamiltonians_by_order(ham, model, conversion_dict, order, R_unitary_part_dict):
    reduced_hamilts = []

    n_qubits = len(list(ham.keys())[0])
    all_qubits = set(range(n_qubits))
    for i in range(len(order)):
        quantum_part = set(order[:i])
        qubits_to_fix_ordered = list(all_qubits.difference(quantum_part))
        H_red = get_reduced_hamiltonian_by_qubits_fixed(ham, model, conversion_dict, qubits_to_fix_ordered, R_unitary_part_dict)
        reduced_hamilts.append(H_red)

    reduced_hamilts.append(ham)
    return reduced_hamilts


def get_wills_order(diagonal_set, order):
    n_q = len(diagonal_set[0])
    order_len = len(order)

    # rectify order
    for i in range(len(order)):
        for j in range(i):
            if order[j] < order[i]:
                order[i] -= 1

    all_qubits = set(range(n_q))
    updated_order = []
    set_previous = {}
    for k in range(order_len + 1):

        z_indices = []
        for d in diagonal_set:
            for i in range(n_q):
                if d[i] == 'Z':
                    z_indices.append(i)
        qubits_quantum_part = all_qubits.difference(set(z_indices))

        qubit_added = qubits_quantum_part.difference(set_previous)
        set_previous = deepcopy(qubits_quantum_part)

        if qubit_added:
            # ignores full noncontextual problem (all z terms fixed)
            updated_order.append(list(qubit_added)[0])

        if order:
            # Drop a qubit:
            i = order[0]
            order.remove(i)
            diagonal_set = diagonal_set[:i] + diagonal_set[i + 1:]

    return updated_order

