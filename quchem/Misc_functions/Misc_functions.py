from scipy.sparse.linalg import expm
from openfermion import qubit_operator_sparse
from scipy.sparse import find
from scipy.sparse import csr_matrix
import numpy as np
from copy import deepcopy
from functools import reduce
import cirq
from collections import namedtuple

def sparse_allclose(A, B, atol=1e-8, rtol=1e-05):
    # https://stackoverflow.com/questions/47770906/how-to-test-if-two-sparse-arrays-are-almost-equal

    if np.array_equal(A.shape, B.shape) == 0:
        raise ValueError('Matrices different shapes!')

    r1, c1, v1 = find(A)  # row indices, column indices, and values of the nonzero matrix entries
    r2, c2, v2 = find(B)

    # # take all the important rows and columns
    # rows = np.union1d(r1, r2)
    # columns = np.union1d(c1, c2)

    # A_check = A[rows, columns]
    # B_check = B[rows, columns]
    # return np.allclose(A_check, B_check, atol=atol, rtol=rtol)

    compare_A_indces = np.allclose(A[r1,c1], B[r1,c1], atol=atol, rtol=rtol)
    if compare_A_indces is False:
        return False

    compare_B_indces = np.allclose(A[r2,c2], B[r2,c2], atol=atol, rtol=rtol)

    if (compare_A_indces is True) and (compare_B_indces is True):
        return True
    else:
        return False


def choose_Pn_index(AC_set_list):
    """
    given a list of anti-commuting operators

    Return index of term with fewest number of change of basis required and fewest Z count

    """
    
    # # size sorting doesn't matter here, as not ansatz or Hamiltonian!
    # sorted_by_size = sorted(AC_set_list, key = lambda x: len(list(zip(*list(*x.terms.keys())))[1]))

   
    ## minimize change of basis
    for ind, op in enumerate(AC_set_list):
        if ind ==0:
            best_ind = 0
            best_Qno, best_Pstrings = zip(*list(*op.terms.keys()))
            best_N_change_basis = sum([1 for sig in best_Pstrings if sig != 'Z'])
            best_Z_count = sum([1 for sig in best_Pstrings if sig == 'Z'])
        else:
            
            Q_Nos, P_strings = zip(*list(*op.terms.keys()))
            N_change_basis = sum([1 for sig in P_strings if sig != 'Z'])

            if (N_change_basis<=best_N_change_basis):
                best_ind = deepcopy(ind)
                best_N_change_basis = deepcopy(N_change_basis)
                best_Z_count = sum([1 for sig in P_strings if sig == 'Z'])
    
     
    ## minimize number of Z terms
    for ind, op in enumerate(AC_set_list):   
        Q_Nos, P_strings = zip(*list(*op.terms.keys()))
        Z_count = sum([1 for sig in P_strings if sig == 'Z'])
        N_change_basis = sum([1 for sig in P_strings if sig != 'Z'])
        
        if (N_change_basis==best_N_change_basis) and (Z_count<best_Z_count): # note keeps best change of basis!
            best_ind = deepcopy(ind)
            best_N_change_basis = deepcopy(N_change_basis)
            best_Z_count= deepcopy(Z_count)
    return  best_ind

def count_circuit_gates(Q_circ):
    oper_list = list(Q_circ.all_operations())
    CNOT_count=0
    two_qubit_gates = 0
    single_qubit_gates = 0 
    for op in oper_list:
        if isinstance(op.gate, cirq.ops.common_gates.CXPowGate):
            CNOT_count+=1
        
        elif len(op.qubits)==2:
            two_qubit_gates+=1
            
        elif len(op.qubits)==1:
            single_qubit_gates+=1
        else:
            print(f'unknown gate: {op}')
    
    gate_count = namedtuple('gate_count', ['single_q', 'CNOT', 'two_q'])
    output = gate_count(single_q=single_qubit_gates, CNOT=CNOT_count, two_q=two_qubit_gates) 
    return output

from cirq.circuits import InsertStrategy
def Get_circuit_depth(Q_circ):
    oper_list = list(Q_circ.all_operations())
    circuit_earliest_possible = cirq.Circuit(oper_list, strategy=InsertStrategy.EARLIEST)
    depth = len(circuit_earliest_possible.moments)
    # print(f'quantum circuit depth: {depth}') 
    return depth


from openfermion import QubitOperator
from openfermion.utils import count_qubits
from functools import reduce
def lexicographical_sort_LADDER_CNOT_cancel(list_P_ops):
    """
    maximises matches of leading terms in Pauliword list (allowing best CNOT cascade cancellation)
    where not possible this function maximise change of basis cancellation
    
    
    ## main idea
    given [Z0 X1 X2 X4] and [Z0 X1 Y2 X4]
    match is [1,1,0,1]
    CNOT_cost = 13
    # note has leading 11 therefore can cancel a CNOT gate
    
    
    given [Z0 Y1 X2 Z4] and [Z0 X1 Y2 X4]
    match is [1,0,0,1]
    CNOT_cost = 9
    # note does NOT have leading 11 therefore CANNOT cancel a CNOT gate
    
    change_of_basis cost = sum(binary) = 2
    2 change of basis gates possible
    
    """
    fullOp = reduce(lambda Op1, Op2: Op1+Op2, list_P_ops)
    max_qubits = count_qubits(fullOp)

    P_Words = []
    for op in list_P_ops:
#         Q_Nos, P_strings = zip(*list(*op.terms.keys()))
#         P_dict = dict(zip(Q_Nos, P_strings)) # zip(keys, values)

        P_dict =  dict(tuple(*op.terms.keys())) 
        arr = [P_dict.get(qNo, 'I') for qNo in range(max_qubits)]
          
        P_Words.append(arr)
    
    P_Words_copy = deepcopy(P_Words)
    re_ordered_ind =[]
    sorted_list = []
    while P_Words!=[]:
        if sorted_list==[]:
            ind_match=0
        else:
            op_prev = sorted_list[-1] # take last sorted term
            
            # get similarity in binary and convert binary to int
            # the larger the int the better the leading match!
            similarity_list = [(op_j, int(''.join((np.array(op_prev)==np.array(op_j)).astype(int).astype(str)), 2)) for op_j in P_Words if op_j != op_i]
            largest_match = max(similarity_list, key=lambda x:x[1])
            
            if np.binary_repr(largest_match[1] ,width=max_qubits)[:2] != '11':
                # no CNOT cancel possible!
                
                ### maximise change of basis cancellation
                # get similarity in binary and sum array
                # the larger the int the better the match between sigma terms!
                similarity_list = [(op_j,sum((np.array(op_prev)==np.array(op_j)).astype(int))) for op_j in P_Words if op_j != op_i]
                largest_match = max(similarity_list, key=lambda x:x[1])
            
            ind_similarity_list = similarity_list.index(largest_match)

            op_j = similarity_list[ind_similarity_list][0]
            ind_match = P_Words.index(op_j)
            
        op_i = P_Words.pop(ind_match)
        sorted_list.append(op_i)
        re_ordered_ind.append(P_Words_copy.index(op_i))

    lex_sorted = (np.array(list_P_ops)[re_ordered_ind]).tolist()
    return lex_sorted

def lexicographical_sort_BASIS_MATCH(list_P_ops):
    """
    maximises adjacent single qubit pauli terms in Pauliword list (allowing best change of basis cancellation)
    """
    fullOp = reduce(lambda Op1, Op2: Op1+Op2, list_P_ops)
    max_qubits = count_qubits(fullOp)

    P_Words = []
    for op in list_P_ops:
        
#         Q_Nos, P_strings = zip(*list(*op.terms.keys()))
#         P_dict = dict(zip(Q_Nos, P_strings)) # zip(keys, values)

        P_dict =  dict(tuple(*op.terms.keys())) 
        arr = [P_dict.get(qNo, 'I') for qNo in range(max_qubits)]
          
        P_Words.append(arr)
    
    P_Words_copy = deepcopy(P_Words)
    re_ordered_ind =[]
    sorted_list = []
    while P_Words!=[]:
        if sorted_list==[]:
            ind_match=0
        else:
            op_prev = sorted_list[-1] # take last sorted term
            
            # get similarity in binary and sum array
            # the larger the int the better the match between sigma terms!
            similarity_list = [(op_j,sum((np.array(op_prev)==np.array(op_j)).astype(int))) for op_j in P_Words if op_j != op_i]
            largest_match = max(similarity_list, key=lambda x:x[1])
            ind_similarity_list = similarity_list.index(largest_match)

            op_j = similarity_list[ind_similarity_list][0]
            ind_match = P_Words.index(op_j)
            
        op_i = P_Words.pop(ind_match)
        sorted_list.append(op_i)
        re_ordered_ind.append(P_Words_copy.index(op_i))


    lex_sorted = (np.array(list_P_ops)[re_ordered_ind]).tolist()
    return lex_sorted




# from qiskit import QuantumCircuit, Aer, execute
# from qiskit.compiler import transpile
# from cirq.contrib.qasm_import import circuit_from_qasm
# def optimized_cirq_circuit_IBM_compiler(cirq_circuit, opt_level=3,
#                                allowed_gates=['id', 'rz', 'ry', 'rx', 'cx' ,'s', 'h', 'y','z'],
#                                 check_optimization = True,
#                                 correct_global_phase=False):
#     """
#     Function that uses IBM's compiler to optimize cirq circuit.
    
#     NOTE this function handles global phase for 0.25 intervals of pi
#     Any other and a modified Zpow gate is used - this may lead to ISSUES, hence a has_modified_Zpow flag is
#     returned!
    
#     Args:
#         cirq_circuit (cirq.Circuit): circuit to optimize
        
#         opt_level (int): level of optimization (see qiskit.compiler.transpile for further details).
#                               0: no optimization
#                               1: light optimization
#                               2: heavy optimization
#                               3: even heavier optimization
        
#         allowed_gates (list): list of strings of allowed qiskit gates
#                                 id: identity
#                                 cx: control x
#                                 sxdg: square root x dagger
#                                 sx: square root x

#                                 example of others:  ['cx', 'u1', 'u2', 'u3']
#                                                      ['id', 'rz', 'sx', 'x', 'cx' ,'s', 'h', 'sxdg','y']
        
#         check_optimization (bool): checks if unitaries of optimized circuit and unoptimized are the same up to global phase

#         correct_global_phase (bool): correct global phase in circuit and check unitaries of circuit before and after opt the SAME
        
#     """
#     # id = identity
    
#     # other possiblities:
#         # ['cx', 'u1', 'u2', 'u3']
    
#     cirq_qasm_file = cirq_circuit.to_qasm(header=False)
    
#     ibm_circuit = QuantumCircuit.from_qasm_str(cirq_qasm_file)
    
#     simplified_circuit = transpile(ibm_circuit,#.reverse_bits(),
#                                    optimization_level=opt_level,
#                                    basis_gates=allowed_gates,
#                                   approximation_degree=1)#no approximation
    
    
# #     global_phase = simplified_circuit.global_phase ## for some reason this seems wrong
# #     print(simplified_circuit.draw())
    
#     ibm_qasm = simplified_circuit.qasm()
#     simplified_cirq_circuit = circuit_from_qasm(ibm_qasm)
    
#     ########
#     ## see matrix_equal function
#     # https://github.com/Qiskit/qiskit-terra/blob/master/qiskit/quantum_info/operators/predicates.py
#     mat1 = cirq_circuit.unitary()
#     for elt in mat1.flat:
#             if abs(elt) > 0:
#                 theta = np.angle(elt)
#                 mat1 = np.exp(-1j * theta) * mat1
#                 break
#     mat2 = simplified_cirq_circuit.unitary()
#     for elt in mat2.flat:
#         if abs(elt) > 0:
#             phi = np.angle(elt)
#             mat2 = np.exp(-1j * phi) * mat2
#             break
#     #######
    
#     if check_optimization:
#         # print(f'phase circuit 1: {theta:.4f}')
#         # print(f'phase circuit 2:  {phi:.4f}')
#         if not np.allclose(mat1, mat2):
#             raise ValueError('circuit compile incorrect (ignores global phase)')
    
#     ### correct global phase
#     has_modified_Zpow = False
#     if correct_global_phase:
#         if not np.isclose(ibm_circuit.global_phase, 0):
#             raise ValueError('global phase issue')
#         else:
#             global_phase = phi
#             if not np.isclose(global_phase, 0):
#                 qubit = list(simplified_cirq_circuit.all_qubits())[0]

#                 if (np.isclose(global_phase, np.pi) or np.isclose(global_phase, -1*np.pi)):
#                     op1 = cirq.rz(2*np.pi).on(qubit)
#                     simplified_cirq_circuit.append(op1)
#                 elif np.isclose(global_phase, -np.pi/2):
#                     op1 = cirq.Z.on(qubit)
#                     op2 = cirq.rz(-np.pi).on(qubit)
#                     simplified_cirq_circuit.append([op1, op2])
#                 elif np.isclose(global_phase, +np.pi/2):
#                     op1 = cirq.Z.on(qubit)
#                     op2 = cirq.rz(+np.pi).on(qubit) 
#                     simplified_cirq_circuit.append([op1, op2])
#                 elif np.isclose(global_phase, np.pi/4):
#                     op1 = (cirq.S**-1).on(qubit)
#                     op2 = cirq.rz(+np.pi/2).on(qubit)
#                     simplied_cirq_circuit.append([op1, op2])
#                 elif np.isclose(global_phase, -np.pi/4):
#                     op1 = (cirq.S).on(qubit)
#                     op2 = cirq.rz(-np.pi/2).on(qubit)
#                     simplified_cirq_circuit.append([op1, op2])
#                 else:
#                     phase_mat = np.eye(2)* np.e**(-1j*global_phase)
#                     Zpow_phase_correction_mat = cirq.ZPowGate(exponent=2, global_shift=-global_phase/(2*np.pi))._unitary_()
#                     if not (np.allclose(Zpow_phase_correction_mat, phase_mat)):
#                         raise ValueError('circuit compile incorrect')
                    
#                     op1 = cirq.ZPowGate(exponent=2, global_shift=-global_phase/(2*np.pi)).on(qubit) # exponent 2 hence divided global phase by 2 (note also divide by pi as in units of pi in ZpowGate definition)
#                     simplified_cirq_circuit.append(op1)
#                     has_modified_Zpow = True
#                     # could use cirq matrix gate instead:
#     #                 print('cannot correct global phase using native gates')
#     #                 print('adding cirq matrix gate')
#     #                 phase_circuit = cirq.Circuit(cirq.MatrixGate(phase_mat).on(qubit))
#     #                 simplified_cirq_circuit.append(list(phase_circuit.all_operations()))
        
#             if check_optimization:
#                 if not np.allclose(cirq_circuit.unitary(), simplified_cirq_circuit.unitary()):
#                     raise ValueError('circuit compile incorrect (includes global phase)')
#     else:
#         global_phase = simplified_circuit.global_phase


#     # convert from named qubit to line qubits
#     qbit_list = [(NamedQ, cirq.LineQubit(int(NamedQ.name[2:]))) for NamedQ in list(simplified_cirq_circuit.all_qubits())]
#     NamedQ_to_LineQ_dict = dict(qbit_list)
#     simplified_cirq_circuit = simplified_cirq_circuit.transform_qubits(lambda x: NamedQ_to_LineQ_dict[x])

#     print('does circuit have modified Zpow:', 'yes'if has_modified_Zpow else 'no')
#     return simplified_cirq_circuit, global_phase, has_modified_Zpow



from qiskit import QuantumCircuit, Aer, execute
from qiskit.compiler import transpile
from cirq.contrib.qasm_import import circuit_from_qasm
def optimized_cirq_circuit_IBM_compiler(cirq_circuit, opt_level=3,
                               allowed_gates=['id', 'rz', 'ry', 'rx', 'cx' ,'s', 'h', 'y','z', 'x'],
                                check_optimization = True):
    """
    Function that uses IBM's compiler to optimize cirq circuit.
    
    NOTE this function handles global phase for 0.25 intervals of pi
    Any other and a modified Zpow gate is used - this may lead to ISSUES, hence a has_modified_Zpow flag is
    returned!
    
    Args:
        cirq_circuit (cirq.Circuit): circuit to optimize
        
        opt_level (int): level of optimization (see qiskit.compiler.transpile for further details).
                              0: no optimization
                              1: light optimization
                              2: heavy optimization
                              3: even heavier optimization
        
        allowed_gates (list): list of strings of allowed qiskit gates
                                id: identity
                                cx: control x
                                sxdg: square root x dagger
                                sx: square root x

                                example of others:  ['cx', 'u1', 'u2', 'u3']
                                                     ['id', 'rz', 'sx', 'x', 'cx' ,'s', 'h', 'sxdg','y']
        
        check_optimization (bool): checks if unitaries of optimized circuit and unoptimized are the same up to global phase

        correct_global_phase (bool): correct global phase in circuit and check unitaries of circuit before and after opt the SAME
        
    """
    
    cirq_qasm_file = cirq_circuit.to_qasm(header=False)
    
    ibm_circuit = QuantumCircuit.from_qasm_str(cirq_qasm_file)
    
    simplified_circuit = transpile(ibm_circuit,#.reverse_bits(),
                                   optimization_level=opt_level,
                                   basis_gates=allowed_gates,
                                  approximation_degree=1)#no approximation
    
    
#     global_phase = simplified_circuit.global_phase ## for some reason this seems wrong
#     print(simplified_circuit.draw())
    
    ibm_qasm = simplified_circuit.qasm()
    simplified_cirq_circuit = circuit_from_qasm(ibm_qasm)
    
    ########
    ## see matrix_equal function
    # https://github.com/Qiskit/qiskit-terra/blob/master/qiskit/quantum_info/operators/predicates.py
    mat1 = cirq_circuit.unitary()
    mat2 = simplified_cirq_circuit.unitary()
    for ind, elt in enumerate(mat1.flat):
            if abs(elt) > 0:
                original_term = elt
                new_term = mat2.flat[ind]

                global_phase = np.angle(original_term/new_term) # find phase difference between unitaries!
                break
    #######
    
    if check_optimization:
        if not np.allclose(cirq_circuit.unitary(), np.exp(1j*global_phase)*simplified_cirq_circuit.unitary()):
            raise ValueError('circuit compile incorrect (ignores global phase)')


    ### correct global phase
    has_modified_Zpow = False
    if not np.isclose(global_phase,0):
        qubit = list(simplified_cirq_circuit.all_qubits())[0]
        op1 = cirq.ZPowGate(exponent=2, global_shift=global_phase/(2*np.pi)).on(qubit) # exponent 2 hence divided global phase by 2 (note also divide by pi as in units of pi in ZpowGate definition)
        simplified_cirq_circuit.append(op1)
        has_modified_Zpow = True

        if check_optimization:
            if not np.allclose(cirq_circuit.unitary(), simplified_cirq_circuit.unitary()):
                raise ValueError('circuit compile incorrect (includes global phase)')

    # convert from named qubit to line qubits
    # note qiskit re-labels from 0 index, therefore need to match qubits by increasing size
    # e.g. if have circuit of exp X2 Y3 ... etc then qiskit will start it from namedqubit 0 NOT two
    # hence need to re-order (done below via sorted lists)
    sorted_original_qubits = sorted(list(cirq_circuit.all_qubits()), key= lambda LineQ: LineQ.x)
    sorted_named_qubits = sorted(list(simplified_cirq_circuit.all_qubits()), key= lambda NamedQ: int(NamedQ.name[2:]))

    NamedQ_to_LineQ_dict = dict(zip(sorted_named_qubits, sorted_original_qubits))
    simplified_cirq_circuit = simplified_cirq_circuit.transform_qubits(lambda x: NamedQ_to_LineQ_dict[x])

    print('does circuit have modified Zpow:', 'yes'if has_modified_Zpow else 'no')
    return simplified_cirq_circuit, global_phase, has_modified_Zpow