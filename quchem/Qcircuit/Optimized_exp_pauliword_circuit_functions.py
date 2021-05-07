import cirq
import numpy as np

from openfermion.utils import count_qubits
from functools import reduce
def ladder_or_stairs_circuit_array(P_Word_list, ladder=True):
    """

    ## example
    from openfermion import QubitOperator
    import numpy as np

    op1 = QubitOperator('Y0 X1', -1j)
    op2 = QubitOperator('Y0 Z1', -1j)
    op3 = QubitOperator('Z0 X1', -1j)
    OP_list = [op1, op2, op3]


    circuit_arr =ladder_circuit_array(OP_list, ladder=True)
    print(circuit_arr)
    
     >> array([['H', 'S', 'C', 'I', 'C', 'Sdag', 'H', 'H', 'S', 'C', 'I', 'C','Sdag', 'H', 'I', 'C', 'I', 'C', 'I'],
               ['I', 'H', 'X', 'R', 'X', 'H',    'I', 'I', 'I', 'X', 'R', 'X', 'I',   'I', 'H', 'X', 'R', 'X', 'H']],
                dtype=object)
    """
    
    change_basis_initial_dict = {
        'X': 'H',
        'Y': 'SH',# S_dag then H (HSdag) 
        'Z': 'I'
    }
    
    change_basis_final_dict = {
        'X': 'H',
        'Y': 'HS',#' H then S (HS)
        'Z': 'I'
    }
    
    
    full_QubitOp = reduce(lambda Op1, Op2: Op1+Op2, P_Word_list)
    max_qubits = count_qubits(full_QubitOp)
    qubits = cirq.LineQubit.range(max_qubits)
    
    empty_circuit_slice = np.array(['I' for _ in range(max_qubits)], dtype=object)
    circuit_array = np.empty((len(qubits),0), dtype=object)
    for ind, P_op in enumerate(P_Word_list):
        PauliStrs, coeff = tuple(*P_op.terms.items())
        qNos, sigmas = zip(*list(PauliStrs))
        
        # initial change basis
        circuit_slice = empty_circuit_slice.copy()
        for qNo, s in zip(qNos, sigmas):
            circuit_slice[qNo] = change_basis_initial_dict[s]
        circuit_array = np.hstack((circuit_array, circuit_slice.reshape(((max_qubits),1))))
            
            
        ### entangle_initial
        entangle_arr = np.empty((len(qubits),0), dtype=object)
        if ladder is True:
            # LADDER approach
            for ind, qNo in enumerate(qNos[:-1]):
                circuit_slice = empty_circuit_slice.copy()
                circuit_slice[qNo] = 'C'
                circuit_slice[qNos[ind+1]] = 'X'
                entangle_arr = np.hstack((entangle_arr, circuit_slice.reshape(((max_qubits),1))))
        else:
            # STAIRS approach
            target_qubit = qNos[-1]
            for ind, qNo in enumerate(qNos[:-1]):
                circuit_slice = empty_circuit_slice.copy()
                circuit_slice[qNo] = 'C'
                circuit_slice[target_qubit] = 'X'
                entangle_arr = np.hstack((entangle_arr, circuit_slice.reshape(((max_qubits),1))))
        circuit_array = np.hstack((circuit_array, entangle_arr))
        ####
        

        ## rotation
        circuit_slice = empty_circuit_slice.copy()
        circuit_slice[qNos[-1]]='R'
        circuit_array = np.hstack((circuit_array, circuit_slice.reshape(((max_qubits),1))))
        
        ## final entangle
        entangle_final = entangle_arr[:,::-1]
        circuit_array = np.hstack((circuit_array, entangle_final))
        
        
        # final change basis
        circuit_slice = empty_circuit_slice.copy()
        for qNo, s in zip(qNos, sigmas):
            circuit_slice[qNo] = change_basis_final_dict[s]
        circuit_array = np.hstack((circuit_array, circuit_slice.reshape(((max_qubits),1))))
        
    return circuit_array

def cancel_common_change_basis(circuit_array):
    """
    Given a circuit array function will cancel any common change of basis gates
    """
    circuit_array=circuit_array.copy()
    for qubit_row in range(circuit_array.shape[0]):
        qubit_operations = circuit_array[qubit_row, :]
        
        for ind, sig in enumerate(qubit_operations):
            if sig in ['I', 'R' 'C']:
                continue

            for new_ind in range(ind+1, circuit_array.shape[1]):
                sig_new = qubit_operations[new_ind]
                if sig_new in ['C', 'X']:
                    break
                elif sig == sig_new[::-1]:
                    qubit_operations[ind]='I'
                    qubit_operations[new_ind]='I'
                    break
                elif sig_new == 'I':
                    continue
                elif sig!=sig_new:
                    break
    return circuit_array

def rewrite_Y_basis_change(circuit_array):
    """
    Re-write anywhere with HS or SH as seperate operations
    """
    
    circuit_array=circuit_array.copy()
    new_circuit_array = np.empty((circuit_array.shape[0],0), dtype=object)
    empty_circuit_slice = np.array(['I' for _ in range(circuit_array.shape[0])], dtype=object)
    
    for col_ind in range(circuit_array.shape[1]):
        term_arr = circuit_array[:, col_ind].copy()
        emplty_slice = empty_circuit_slice.copy()
        
        if 'S' in ''.join(term_arr):
            SH = np.where(term_arr=='SH')[0]
            HS = np.where(term_arr=='HS')[0]
            
            term_arr[SH]= 'Sdag'
            emplty_slice[SH]='H'
            
            term_arr[HS]= 'H'
            emplty_slice[HS]='S'
            
            
            new_slice = np.vstack((term_arr, emplty_slice)).T
        else:
            new_slice = term_arr.reshape((circuit_array.shape[0],1))

        new_circuit_array = np.hstack((new_circuit_array, new_slice))
        
    return new_circuit_array

def cancel_S_gates_through_controls(circuit_array):
    circuit_array=circuit_array.copy()
    for qubit_row in range(circuit_array.shape[0]):
        qubit_operations = circuit_array[qubit_row, :]
        
        for ind, sig in enumerate(qubit_operations):
            if sig not in ['S', 'Sdag']:
                continue
            for new_ind in range(ind+1, circuit_array.shape[1]):
                sig_new = qubit_operations[new_ind]
                if sig_new in ['X', 'H', 'R']:
                    # cannot move through these gates
                    break
                elif sig_new in ['I','C']:
                    # S gates commute with controls
                    continue
                elif (sig == 'S') and (sig_new == 'Sdag'):
                    qubit_operations[ind]='I'
                    qubit_operations[new_ind]='I'
                    break
                elif (sig == 'Sdag') and (sig_new == 'S'):
                    qubit_operations[ind]='I'
                    qubit_operations[new_ind]='I'
                    break
                else:
                    raise ValueError(f'unkown opertion:{sig, sig_new}')
    return circuit_array


def cancel_CNOT_Gates(circuit_array):
    circuit_array=circuit_array.copy()
    for qubit_row in range(circuit_array.shape[0]-1):
        qubit_operations = circuit_array[qubit_row, :]
        
        for ind, sig_control in enumerate(qubit_operations):
            if sig_control != 'C':
                continue
            
            next_row_ind = np.where(circuit_array[:,ind]=='X')[0][0]
            qubit_operations_NEXT = circuit_array[next_row_ind, :]
            
            sig_target = qubit_operations_NEXT[ind]
            if sig_target != 'X':
                continue
            
            for new_ind in range(ind+1, circuit_array.shape[1]):
                sig_new_CONTROL = qubit_operations[new_ind]
                sig_new_Target = qubit_operations_NEXT[new_ind]
                
                if (sig_new_CONTROL in ['I', 'S', 'Sdag'] and sig_new_Target in ['I']):
                    continue
                elif sig_new_CONTROL !='C':
                    break
                elif sig_new_Target !='X':
                    break
                else:
                    qubit_operations[ind]='I'
                    qubit_operations_NEXT[ind]='I'
                    qubit_operations[new_ind]='I'
                    qubit_operations_NEXT[new_ind]='I'
                    break
    return circuit_array

def YX_simplification(circuit_array):
    """
    Given a circuit array will replace any HSH terms with X^0.5 and H S^-0.5 H with X^-0.5
    """
    circuit_array=circuit_array.copy()
    for qubit_row in range(circuit_array.shape[0]):
        qubit_operations = circuit_array[qubit_row, :]
        
        for ind_i, sig in enumerate(qubit_operations):
            if sig !='H':
                continue
                
            for ind_j in range(ind_i+1, circuit_array.shape[1]):
                sig_new = qubit_operations[ind_j]
                if sig_new in ['C', 'X', 'R', 'H']:
                    break
                elif sig_new == 'I':
                    continue
                elif sig_new in ['S', 'Sdag']:
                    for ind_k in range(ind_j+1, circuit_array.shape[1]):
                        sig_k = qubit_operations[ind_k]                        
                        if sig_k in ['R', 'C', 'Sdag', 'S', 'X']:
                            break
                        elif sig_k == 'I':
                            continue
                        elif sig_k =='H':
                            qubit_operations[ind_i]='I'
                            qubit_operations[ind_j]='I'
                            if sig_new == 'S':
                                qubit_operations[ind_k]='Xpow_pos_half'
                            elif sig_new == 'Sdag':
                                qubit_operations[ind_k]='Xpow_neg_half'
                            else:
                                raise ValueError(f'error in j operation {sig_new}')
                            break
                        else:
                            raise ValueError(f'unknown k operation {sig_k}')
                    break
                else:
                    raise ValueError(f'unknown j operation {sig_new}')
    return circuit_array

def circuit_array_to_circuit(circuit_array, angle_list, P_Word_list):
    
    qubits = cirq.LineQubit.range(circuit_array.shape[1])
    circuit = cirq.Circuit()
    
    angle_ind=0
    for circuit_slice in circuit_array.T:
        non_I_indices = np.where(circuit_slice!='I')[0]

        if 'C' in circuit_slice:
            circuit.append(cirq.CNOT(qubits[non_I_indices[0]], qubits[non_I_indices[1]]))
        else:
            for qNo in non_I_indices:
                op = circuit_slice[qNo]
                if op == 'H':
                    circuit.append(cirq.H(qubits[qNo]))
                elif op == 'R':
                    Pword = P_Word_list[angle_ind]
                    complex_coeff = list(Pword.terms.values())[0]
                    if complex_coeff.imag==1:
                        circuit.append(cirq.rz(-2*angle_list[angle_ind]).on(qubits[qNo]))
                    elif complex_coeff.imag==-1:
                        circuit.append(cirq.rz(2*angle_list[angle_ind]).on(qubits[qNo]))
                    else:
                        raise ValueError(f'PauliWord not defined with correct phase: {complex_coeff}')
                    angle_ind+=1
                elif op == 'S':
                    circuit.append(cirq.S(qubits[qNo]))
                elif op == 'Sdag':
                    circuit.append((cirq.S**-1).on(qubits[qNo]))
                elif op == 'Xpow_pos_half':
                    circuit.append((cirq.X**0.5).on(qubits[qNo]))
                elif op == 'Xpow_neg_half':
                    circuit.append((cirq.X**-0.5).on(qubits[qNo]))
                elif op == 'X':
                    continue
                else:
                    print(f'unknown operation: {op}')
    return circuit
    
    

from scipy.sparse.linalg import expm
from openfermion import qubit_operator_sparse
from openfermion.utils import count_qubits
def Optimized_LADDER_circuit(P_Word_list, angle_list, check_reduction=False):

    circuit_array=ladder_or_stairs_circuit_array(P_Word_list, ladder=True)
    circuit_array_basis_cancelled=cancel_common_change_basis(circuit_array)
    expanded_circuit_array = rewrite_Y_basis_change(circuit_array_basis_cancelled)
    circuit_array_S_cancelled=cancel_S_gates_through_controls(expanded_circuit_array)
    circuit_array_basis_cancelled_2=cancel_common_change_basis(circuit_array_S_cancelled)
    circuit_array_CNOT_cancel=cancel_CNOT_Gates(circuit_array_basis_cancelled_2)
    circuit_XY_cancel=YX_simplification(circuit_array_CNOT_cancel)
    opt_circuit = circuit_array_to_circuit(circuit_XY_cancel, angle_list, P_Word_list)

    if check_reduction:
        fullOp = reduce(lambda Op1, Op2: Op1+Op2, P_Word_list)
        N_qubits = count_qubits(fullOp)
        # N_qubits = max([lineQ.x for lineQ in list(opt_circuit.all_qubits())]) + 1
        mat_list = [expm(qubit_operator_sparse(op, n_qubits = N_qubits)*angle_list[ind]).todense() for ind, op in enumerate(P_Word_list)]
        lin_alg_mat = reduce(np.dot, mat_list[::-1])

        if not np.allclose(lin_alg_mat, opt_circuit.unitary(qubits_that_should_be_present=cirq.LineQubit.range(N_qubits))):
            raise ValueError('circuit reduction incorrect!') 

    return opt_circuit

def Optimized_STAIRS_circuit(P_Word_list, angle_list, check_reduction=False):

    circuit_array=ladder_or_stairs_circuit_array(P_Word_list, ladder=False)
    circuit_array_basis_cancelled=cancel_common_change_basis(circuit_array)
    expanded_circuit_array = rewrite_Y_basis_change(circuit_array_basis_cancelled)
    circuit_array_S_cancelled=cancel_S_gates_through_controls(expanded_circuit_array)
    circuit_array_basis_cancelled=cancel_common_change_basis(circuit_array_S_cancelled)
    circuit_array_CNOT_cancel=cancel_CNOT_Gates(circuit_array_basis_cancelled)
    circuit_XY_cancel=YX_simplification(circuit_array_CNOT_cancel)
    opt_circuit =circuit_array_to_circuit(circuit_XY_cancel, angle_list, P_Word_list)

    if check_reduction:
        N_qubits = max([lineQ.x for lineQ in list(opt_circuit.all_qubits())]) + 1
        mat_list = [expm(qubit_operator_sparse(op, n_qubits = N_qubits)*angle_list[ind]).todense() for ind, op in enumerate(P_Word_list)]
        lin_alg_mat = reduce(np.dot, mat_list[::-1])

        if not np.allclose(lin_alg_mat, opt_circuit.unitary()):
            raise ValueError('circuit reduction incorrect!') 

    return opt_circuit


#########
# # gate
# def change_basis_initial(single_pauli_str, line_qubit):
#     if single_pauli_str == 'X':
#         return cirq.H.on(line_qubit)
#     elif single_pauli_str == 'Y':
#         # return cirq.rx(np.pi / 2).on(line_qubit)
#         return [cirq.H.on(line_qubit), cirq.S.on(line_qubit)]
#     elif (single_pauli_str == 'Z') or (single_pauli_str == 'I'):
#         return None
#     elif single_pauli_str == 'XY':
#         return cirq.S.on(line_qubit)
#     elif single_pauli_str == 'YX':
#         return (cirq.S**-1).on(line_qubit)
#     else:
#         raise ValueError("Qubit Operation: {} is NOT a Pauli operation".format(single_pauli_str))


# def change_basis_final(single_pauli_str, line_qubit):

#     if single_pauli_str == 'X':
#         return cirq.H.on(line_qubit)
#     elif single_pauli_str == 'Y':
#         return [(cirq.S**-1).on(line_qubit), cirq.H.on(line_qubit)]
#     elif (single_pauli_str == 'Z') or (single_pauli_str == 'I'):
#         return None
#     elif single_pauli_str == 'XY':
#         # H . H S ==> S
#         return cirq.S.on(line_qubit)
#     elif single_pauli_str == 'YX':
#         # S_dagger H . H ==> S_dagger
#         return (cirq.S**-1).on(line_qubit)
#     else:
#         raise ValueError("Qubit Operation: {} is NOT a Pauli operation".format(single_pauli_str))

# from openfermion.utils import count_qubits
# from functools import reduce
# def exp_pauliword_reduced_QC_STAIRS(P_A, theta_A, pre_P_A_term=None, post_P_A_term=None):
#     """

#     ## example
#     from openfermion import QubitOperator
#     import numpy as np

#     op1 = QubitOperator('X0 X1 X2', -1j)
#     op2 = QubitOperator('Z0 X1 X2', -1j)
#     op3 = QubitOperator('X0 Y1 X2', -1j)
    
#     OP_list = [op1, op2, op3]
#     theta_A= np.pi

#     circuit_Test = Pair_exp_pauliword_STAIRS(OP_list[1], 
#                                          theta_A,
#                                          pre_P_A_term=None,
#                                          post_P_A_term=None)
#     print(circuit_Test)
                                   
#         0: ───────@─────────────────────@─────────
#                   │                     │
#         1: ───H───┼───@────────────@────┼─────H───
#                   │   │            │    │
#         2: ───H───X───X───Rz(2π)───X────X─────H───



#     circuit_Test_REDUCED = Pair_exp_pauliword_STAIRS(OP_list[1], 
#                                          theta_A,
#                                          pre_P_A_term=OP_list[0],
#                                          post_P_A_term=OP_list[2])
#     print(circuit_Test_REDUCED)
            
#         0: ───@─────────────────@─────
#               │                 │
#         1: ───┼────────────@────┼───H─
#               │            │    │
#         2: ───X───Rz(2π)───X────X─────
                              
                                   
#     """

#     input_list = [P_A, pre_P_A_term, post_P_A_term]
#     active_list = [term for term in input_list if term is not None]
    
#     full_QubitOp = reduce(lambda Op1, Op2: Op1+Op2, active_list)
#     max_qubits = count_qubits(full_QubitOp)
#     qubits = cirq.LineQubit.range(max_qubits)
    
    
#     PauliStrs_A, coeff_A = tuple(*P_A.terms.items())
#     qNos_A, sigmas_A = zip(*list(PauliStrs_A))

#     circuit = cirq.Circuit()

#     if pre_P_A_term is None:
#         ## no cancellation on LHS for P_A

#         # change of basis
#         for qNo, sigma_str in zip(qNos_A, sigmas_A):
#             gate = change_basis_initial(sigma_str, qubits[qNo])
#             if gate: circuit.append(gate) 

#         ## entangle initial for P_A
#         target_q = qNos_A[-1]
#         for ind, qNo in enumerate(qNos_A[:-1]):
#             circuit.append(cirq.CNOT.on(qubits[qNo], qubits[target_q]))

#     else:
#         ## check for cancellations on LHS for P_A
#         PauliStrs_pre, coeff_pre = tuple(*pre_P_A_term.terms.items())
#         qNos_pre, sigmas_pre = zip(*list(PauliStrs_pre))

#         common_qubits = np.intersect1d(qNos_pre, qNos_A)
        
#         reduction_possible_dict = {common_q: sigmas_pre[qNos_pre.index(common_q)]==sigmas_A[qNos_A.index(common_q)]
#                                                             for common_q in common_qubits}
        
#         for qNo, sigma_str in zip(qNos_A, sigmas_A):
#             if reduction_possible_dict.get(qNo, False):
#                 pass
#             else:
#                 gate = change_basis_initial(sigma_str, qubits[qNo])
#                 if gate: circuit.append(gate) 
                    
#         ## entangle initial for P_A
#         target_q = qNos_A[-1]
        
#         pre_target_qubit  = qNos_pre[-1]
#         pre_CNOT_red= reduction_possible_dict.get(target_q, True)
#         if pre_target_qubit != target_q:
#             for ind, qNo in enumerate(qNos_A[:-1]):
#                 circuit.append(cirq.CNOT.on(qubits[qNo], qubits[target_q]))
#         elif pre_CNOT_red:
#             for ind, qNo in enumerate(qNos_A[:-1]):
#                 if reduction_possible_dict.get(qNo, False):
#                     continue
#                 else:
#                     circuit.append(cirq.CNOT.on(qubits[qNo], qubits[target_q]))
#         else:
#             for ind, qNo in enumerate(qNos_A[:-1]):
#                 circuit.append(cirq.CNOT.on(qubits[qNo], qubits[target_q]))


#     ### PERFORM ROTATION
#     if coeff_A.imag==0:
#         raise ValueError('not valid qubit rotation')
#     elif coeff_A.imag < 0:
#         circuit.append(cirq.rz(2 * theta_A * np.abs(coeff_A.imag)).on(qubits[qNos_A[-1]]))
#     else:
#         circuit.append(cirq.rz(-2 * theta_A * np.abs(coeff_A.imag)).on(qubits[qNos_A[-1]]))


#     #### RHS

#     if post_P_A_term is None:
#         ## no cancellation on RHS for P_A

#         ## final initial for P_A
#         target_q = qNos_A[-1]
#         for ind, qNo in enumerate(qNos_A[:-1][::-1]):
#             circuit.append(cirq.CNOT.on(qubits[qNo], qubits[target_q]))
            
#         # change of basis final
#         for qNo, sigma_str in zip(qNos_A, sigmas_A):
#             gate = change_basis_final(sigma_str, qubits[qNo])
#             if gate: circuit.append(gate) 

#     else:
#         ## check for cancellations on RHS for P_A
#         PauliStrs_post, coeff_post = tuple(*post_P_A_term.terms.items())
#         qNos_post, sigmas_post = zip(*list(PauliStrs_post))

#         common_qubits = np.intersect1d(qNos_post, qNos_A)

#         reduction_possible_dict = {common_q: sigmas_post[qNos_post.index(common_q)]==sigmas_A[qNos_A.index(common_q)]
#                                                             for common_q in common_qubits}
        
        
#         ## entangle final for P_A
#         target_q = qNos_A[-1]
        
#         post_CNOT_red= reduction_possible_dict.get(target_q, True)
#         post_target_qubit  = qNos_post[-1]
#         if target_q != post_target_qubit:
#             for ind, qNo in enumerate(qNos_A[:-1][::-1]):
#                 circuit.append(cirq.CNOT.on(qubits[qNo], qubits[target_q]))

#         elif post_CNOT_red:
#             for ind, qNo in enumerate(qNos_A[:-1][::-1]):
#                 if reduction_possible_dict.get(qNo, False):
#                     continue
#                 else:
#                     circuit.append(cirq.CNOT.on(qubits[qNo], qubits[target_q]))
#         else:
#             for ind, qNo in enumerate(qNos_A[:-1][::-1]):
#                 circuit.append(cirq.CNOT.on(qubits[qNo], qubits[target_q]))

#         # change of basis
#         for qNo, sigma_str in zip(qNos_A, sigmas_A):
#             if reduction_possible_dict.get(qNo, False):
#                 pass
#             else:
#                 gate = change_basis_final(sigma_str, qubits[qNo])
#                 if gate: circuit.append(gate) 
                    

#     return circuit


# def exp_pauliword_reduced_QC_LADDER(P_A, theta_A, pre_P_A_term=None, post_P_A_term=None):
#     """

#     ## example
#     from openfermion import QubitOperator
#     import numpy as np

#     op1 = QubitOperator('X0 Y1 X2', -1j)
#     op2 = QubitOperator('X0 X1 X2', -1j)
#     op3 = QubitOperator('Y0 Y1 X2', -1j)

#     OP_list = [op1, op2, op3]
#     theta_A= np.pi 


#     circuit =exp_pauliword_reduced_QC_LADDER(op1 , theta_A, pre_P_A_term=op2, post_P_A_term=op3)
#     print(circuit)
    
#     0: ───────────────────────────────@───H───
#                                       │
#     1: ───Rx(0.5π)───@────────────@───X───────
#                      │            │
#     2: ──────────────X───Rz(2π)───X───────────
    
#     """

#     input_list = [P_A, pre_P_A_term, post_P_A_term]
#     active_list = [term for term in input_list if term is not None]
    
#     full_QubitOp = reduce(lambda Op1, Op2: Op1+Op2, active_list)
#     max_qubits = count_qubits(full_QubitOp)
#     qubits = cirq.LineQubit.range(max_qubits)
    
    
#     PauliStrs_A, coeff_A = tuple(*P_A.terms.items())
#     qNos_A, sigmas_A = zip(*list(PauliStrs_A))

#     circuit = cirq.Circuit()

#     if pre_P_A_term is None:
#         ## no cancellation on LHS for P_A

#         # change of basis
#         for qNo, sigma_str in zip(qNos_A, sigmas_A):
#             gate = change_basis_initial(sigma_str, qubits[qNo])
#             if gate: circuit.append(gate) 

#         ## entangle initial for P_A
#         for ind, qNo in enumerate(qNos_A[:-1]):
#             circuit.append(cirq.CNOT.on(qubits[qNo], qubits[qNos_A[ind+1]]))

#     else:
#         ## check for cancellations on LHS for P_A
#         PauliStrs_pre, coeff_pre = tuple(*pre_P_A_term.terms.items())
#         qNos_pre, sigmas_pre = zip(*list(PauliStrs_pre))

#         common_qubits = np.intersect1d(qNos_pre, qNos_A)
        
#         change_B_possible_dict = {common_q: sigmas_pre[qNos_pre.index(common_q)]==sigmas_A[qNos_A.index(common_q)]
#                                                             for common_q in common_qubits}
        
#         for qNo, sigma_str in zip(qNos_A, sigmas_A):
#             if change_B_possible_dict.get(qNo, False):
#                 pass
#             else:
#                 gate = change_basis_initial(sigma_str, qubits[qNo])
#                 if gate: circuit.append(gate) 
                    
#         ## entangle initial for P_A
#         CNOT_reduction_flag=True
#         for ind, qNo in enumerate(qNos_A[:-1]):
#             next_ent_qubit = qNos_A[ind+1]
#             if (change_B_possible_dict.get(qNo, False) and CNOT_reduction_flag and change_B_possible_dict.get(next_ent_qubit, False)):
#                 continue
#             else:
#                 CNOT_reduction_flag = False
#                 circuit.append(cirq.CNOT.on(qubits[qNo], qubits[next_ent_qubit]))


#     ### PERFORM ROTATION
#     if coeff_A.imag==0:
#         raise ValueError('not valid qubit rotation')
#     elif coeff_A.imag < 0:
#         circuit.append(cirq.rz(2 * theta_A * np.abs(coeff_A.imag)).on(qubits[qNos_A[-1]]))
#     else:
#         circuit.append(cirq.rz(-2 * theta_A * np.abs(coeff_A.imag)).on(qubits[qNos_A[-1]]))


#     #### RHS

#     if post_P_A_term is None:
#         ## no cancellation on RHS for P_A

#         ## final initial for P_A
#         reverse_order = qNos_A[::-1]
#         for ind, qNo in enumerate(reverse_order[:-1]):
#             circuit.append(cirq.CNOT.on(qubits[reverse_order[ind+1]], qubits[qNo]))
            
#         # change of basis final
#         for qNo, sigma_str in zip(qNos_A, sigmas_A):
#             gate = change_basis_final(sigma_str, qubits[qNo])
#             if gate: circuit.append(gate) 

#     else:
#         ## check for cancellations on RHS for P_A
#         PauliStrs_post, coeff_post = tuple(*post_P_A_term.terms.items())
#         qNos_post, sigmas_post = zip(*list(PauliStrs_post))

#         common_qubits = np.intersect1d(qNos_post, qNos_A)

#         reduction_possible_dict = {common_q: sigmas_post[qNos_post.index(common_q)]==sigmas_A[qNos_A.index(common_q)]
#                                                             for common_q in common_qubits}
        
        
#         temp_circuit = cirq.Circuit()
#         for qNo, sigma_str in zip(qNos_A, sigmas_A):
#             if reduction_possible_dict.get(qNo, False):
#                 pass
#             else:
#                 gate = change_basis_initial(sigma_str, qubits[qNo]) #inital used as will mirror
#                 if gate: temp_circuit.append(gate) 

#         ## final initial for P_A
#         CNOT_reduction_flag=True
#         for ind, qNo in enumerate(qNos_A[:-1]):
#             next_ent_qubit = qNos_A[ind+1]
#             if (reduction_possible_dict.get(qNo, False) and CNOT_reduction_flag and reduction_possible_dict.get(next_ent_qubit, False)):
#                 continue
#             else:
#                 CNOT_reduction_flag = False
#                 temp_circuit.append(cirq.CNOT.on(qubits[qNo], qubits[qNos_A[ind+1]]))
        
#         mirror_circuit = cirq.inverse(temp_circuit) # inverse circuit!
#         del temp_circuit
#         circuit.append(mirror_circuit)
#     return circuit



# from openfermion.linalg import qubit_operator_sparse
# from scipy.sparse.linalg import expm
# def Circuit_from_list_PauliWords_STAIRS(list_PauliWords, list_angles, check_reduction=False):
#     """
#     generate circuit from list of exp pauliwords.

#     Note list_PauliWords should be pre-ordered to maximize reductions! (lexographical order) 
    

#     ## example
#     from openfermion import QubitOperator
#     op1 = QubitOperator('X0 X1 X2', -1j)
#     op2 = QubitOperator('Z0 X1 X2', -1j)
#     op3 = QubitOperator('X0 Y1 Y2', -1j)
    
#     OP_list = [op1, op2, op3]

#     """
#     if len(list_PauliWords)!=  len(list_angles):
#         raise ValueError('Incorrect number of PauliWords/angles')

#     circuit = cirq.Circuit()

#     for ind, Pop in enumerate(list_PauliWords):
#         if ind ==0:
#             c=exp_pauliword_reduced_QC_STAIRS(Pop, list_angles[ind], pre_P_A_term=None, post_P_A_term=list_PauliWords[ind+1])
#         elif ind ==(len(list_PauliWords)-1):
#             c=exp_pauliword_reduced_QC_STAIRS(Pop, list_angles[ind], pre_P_A_term=list_PauliWords[ind-1], post_P_A_term=None)
#         else:
#             c= exp_pauliword_reduced_QC_STAIRS(Pop, list_angles[ind], pre_P_A_term=list_PauliWords[ind-1], post_P_A_term=list_PauliWords[ind+1])
        
#         circuit.append(c)

#     if check_reduction:
#         N_qubits = max([lineQ.x for lineQ in list(circuit.all_qubits())]) + 1
#         mat_list = [expm(qubit_operator_sparse(op, n_qubits = N_qubits)*list_angles[ind]).todense() for ind, op in enumerate(list_PauliWords)]
#         lin_alg_mat = reduce(np.dot, mat_list[::-1])

#         if not np.allclose(lin_alg_mat, circuit.unitary()):
#             raise ValueError('circuit reduction incorrect!') 

#     return circuit

# def Circuit_from_list_PauliWords_LADDER(list_PauliWords, list_angles, check_reduction=False):
#     """
#     generate circuit from list of exp pauliwords.

#     Note list_PauliWords should be pre-ordered to maximize reductions! (lexographical order) 
    

#     ## example
#     from openfermion import QubitOperator
#     op1 = QubitOperator('X0 X1 X2', -1j)
#     op2 = QubitOperator('Z0 X1 X2', -1j)
#     op3 = QubitOperator('X0 Y1 Y2', -1j)
    
#     OP_list = [op1, op2, op3]

#     """
#     if len(list_PauliWords)!=  len(list_angles):
#         raise ValueError('Incorrect number of PauliWords/angles')

#     circuit = cirq.Circuit()

#     for ind, Pop in enumerate(list_PauliWords):
#         if ind ==0:
#             c=exp_pauliword_reduced_QC_LADDER(Pop, list_angles[ind], pre_P_A_term=None, post_P_A_term=list_PauliWords[ind+1])
#         elif ind ==(len(list_PauliWords)-1):
#             c=exp_pauliword_reduced_QC_LADDER(Pop, list_angles[ind], pre_P_A_term=list_PauliWords[ind-1], post_P_A_term=None)
#         else:
#             c= exp_pauliword_reduced_QC_LADDER(Pop, list_angles[ind], pre_P_A_term=list_PauliWords[ind-1], post_P_A_term=list_PauliWords[ind+1])
        
#         circuit.append(c)

#     if check_reduction:
#         N_qubits = max([lineQ.x for lineQ in list(circuit.all_qubits())]) + 1
#         mat_list = [expm(qubit_operator_sparse(op, n_qubits = N_qubits)*list_angles[ind]).todense() for ind, op in enumerate(list_PauliWords)]
#         lin_alg_mat = reduce(np.dot, mat_list[::-1])

#         if not np.allclose(lin_alg_mat, circuit.unitary()):
#             raise ValueError('circuit reduction incorrect!') 

#     return circuit

