import numpy as np
import scipy as sp
from scipy.optimize import minimize_scalar
from scipy.sparse import coo_matrix
import itertools
from functools import reduce
import random
from datetime import datetime
from datetime import timedelta
from copy import deepcopy
from openfermion.utils import hermitian_conjugated
from openfermion import qubit_operator_sparse
from openfermion.linalg import qubit_operator_sparse
from openfermion import hermitian_conjugated
from openfermion.ops import QubitOperator


import cs_vqe as c
import quchem.Misc_functions.conversion_scripts as conv_scr 
from quchem.Unitary_Partitioning.Unitary_partitioning_Seq_Rot import Get_Xsk_op_list

def diagonalize_epistemic_SeqRot(model,fn_form,ep_state, check_reduction=False):
    
    assert(len(ep_state[0]) == fn_form[0])
    assert(len(model[0]) == fn_form[0])
    assert(len(ep_state[1]) == fn_form[1])
    assert(len(model[1]) == fn_form[1])
    
    rotations = []
    R_SeqRot = None
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
                rotations.append( ['pi/2', K] )
                
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
    
    return R_sk_OP_list, rotations, GuA, np.real(ep_state_trans)


# Given a Hamiltonian ham, for which the noncontextual part has a quasi-quantized model
# specified by model and fn_form, and a noncontextual ground state specified by ep_state,
# returns the noncontextual ground state energy plus the quantum correction.
def quantum_correction_SeqRot(ham,model,fn_form,ep_state, check_reduction=False):
    
    R_SeqRot, rotations, diagonal_set, vals = diagonalize_epistemic_SeqRot(model,fn_form,ep_state, check_reduction=check_reduction)
    
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

