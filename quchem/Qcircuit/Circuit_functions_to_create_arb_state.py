import cirq
import numpy as np
from copy import deepcopy
from quchem.Qcircuit.misc_quantum_circuit_functions import Get_state_as_str

# import sympy as symp
# def Single_qubit_rotation(zero_state_amp, one_state_amp):
#     """
#     Get rotation to create passed in qubit state
    
#     |psi> = cos(theta/2) |0> + e^{-1j * phi/2} sin(theta/2) |1>
    
#     See equation 8 and 9 of https://arxiv.org/pdf/quant-ph/0406176v5.pdf
    
#     """
#     zero_state_amp =np.complex128(zero_state_amp)
#     one_state_amp = np.complex128(one_state_amp)
    
#     norm = np.sqrt(np.abs(zero_state_amp)**2 + np.abs(one_state_amp)**2 )

#     if np.isclose(norm, 0):
#         theta = 0
#         a_arg = 0
#         b_arg = 0
#         final_t = 0
#         phi = 0
#     else:
#         theta=2*symp.acos(np.abs(zero_state_amp) / norm)
        
#         a_arg = symp.atan2(zero_state_amp.imag, zero_state_amp.real)
#         b_arg = symp.atan2(one_state_amp.imag, one_state_amp.real)
#         final_t = a_arg + b_arg
#         phi = b_arg - a_arg
    
#     return (norm * symp.exp(1.J * final_t / 2)).evalf(), theta.evalf(), phi.evalf()

def Single_qubit_rotation(zero_state_amp, one_state_amp):
    """
    Get rotation to create passed in  single qubit state. Inputs are 𝛼0 and 𝛼1 of:
    
    |𝜓⟩= 𝛼0|0⟩+𝛼1|1⟩ = re^{𝑖𝑡/2} * [ e^{−𝑖𝜙/2} cos(𝜃/2)|0⟩ +e^{+𝑖𝜙/2} sin(𝜃/2)|1⟩ ]
    
    See https://arxiv.org/pdf/quant-ph/0406176v5.pdf

    Args:
        zero_state_amp (complex): complex amplitude of |0⟩
        one_state_amp (complex): complex amplitude of |1⟩

    Returns:
        global_phase: constant factor of re^{it/2}
        theta (float) : angle 
        phi (float) : angle 

    
    See equation 8 and 9 of https://arxiv.org/pdf/quant-ph/0406176v5.pdf

    ** Example **

    import numpy
    alpha_0 = 1/np.sqrt(2) 
    alpha_1 = 1j/np.sqrt(2)
    
    global_phase , theta, phi = Single_qubit_rotation(alpha_0, alpha_1)
    
    """
    zero_state_amp =np.complex128(zero_state_amp)
    one_state_amp = np.complex128(one_state_amp)
    
    norm = np.sqrt(np.abs(zero_state_amp)**2 + np.abs(one_state_amp)**2 )
    
    if np.isclose(norm, 0):
        theta = 0
        a_arg = 0
        b_arg = 0
        final_t = 0
        phi = 0
    else:
        theta= 2 * np.arccos(np.abs(zero_state_amp) / norm)
        # theta=2*np.arctan2(np.abs(one_state_amp)/ norm.real, np.abs(zero_state_amp)/ norm.real)

        a_arg = np.angle(zero_state_amp)
        b_arg = np.angle(one_state_amp)
        final_t = a_arg + b_arg
        phi = b_arg - a_arg
    
    global_phase = norm * np.exp(1.J * final_t / 2)

    return global_phase, theta, phi

def Rotations_to_disentangle(qubit_state_vector):
    """
    pg 11 of https://arxiv.org/pdf/quant-ph/0406176v5.pdf
    
    Method to work out Ry and Rz rotation angles to disentangle the least significant bit (LSB).
    These rotations make up a block diagonal matrix U == multiplexor
    
    
    
    ### futher background:
    
    Given |ψ> an (n+1) qubit state, seperate the state into a separable (unentanged) state by the following circuit:
    
               n: ─/──(C)─────────────(C)────────────── |𝜓'⟩ (n-qubits)
     |𝜓⟩ (n+1)         │               │              
               1: ────Rz(phi_ZERO)───Ry(theta_ZERO)──── |0⟩


    the 2^{n+1} state vector is split into TWO 2^{n} states... This can be done by the circuit above.
    
    Overall |𝜓⟩ ( 2^{n+1} state vector) is split into 2^{n} contiguous 2-element blocks. This can be interpreted as a
    2D complex vector. We can label this |ψ_{c}>
    
    Then:
    
    Rz(-φ_{c}) Ry(-θ_{c}) |ψ> = r_{c} exp(1j*t_{c}) |0 >
    
    |ψ''> is the n-qubit state given by the 2^{n}-element row vector with c-th entry r_{c} exp(1j*t_{c}).
    
    If we let U be the block diagonal sum ⊕_{c} Ry(-θ_{c}) Rz(-φ_{c}). THEN:
    
                𝑈|𝜓⟩=|𝜓'⟩⊗|0⟩
    
    We can implement U as a multiplexed Rz gate followed by a multiplexed Ry gate! The unitary to do this is:

                    [[Ry(theta_1).Rz(phi_1)          0             ...    0   ],
                     [         0             Ry(theta_2).Rz(phi_2) ...    0   ],
                𝑈 =  [         .         .       ..                       0   ],
                     [         .         .            ..           ...    0
                     [         0         0                         ... Ry(theta_2^n).Rz(phi_2^n)]]
    
    
    Example:

    psi = [0.5, 0.5, 0.5, 0.5]

    remaining_vector, theta_list, phi_list = Rotations_to_disentangle(psi)

    print(theta_list)
    >> [-1.5707963267948968, -1.5707963267948968]

    print(phi_list)
    >> [-0.0, -0.0]

    print(remaining_vector)
    >> [(0.7071067811865476+0j), (0.7071067811865476+0j)]

              ┌                                           ┐
              │ 0.5  +0.j -0.289+0.j -0.408+0.j -0.707+0.j│
        0: ───│ 0.5  +0.j  0.866+0.j  0.   +0.j  0.   +0.j│───(0)─────@───────(0)─────────@─────────── |+⟩
              │ 0.5  +0.j -0.289+0.j  0.816+0.j  0.   +0.j│   │       │       │           │
              │ 0.5  +0.j -0.289+0.j -0.408+0.j  0.707+0.j│   │       │       │           │
              └                                           ┘   │       │       │           │
              │                                               │       │       │           │
        1: ───#2──────────────────────────────────────────────Rz(0)───Rz(0)───Ry(-0.5π)───Ry(-0.5π)─── |0⟩

    """
    
    remaining_vector = []
    theta_list = []
    phi_list = []
    
    param_len = len(qubit_state_vector)
    for state_ind in range(param_len//2):
            # Ry and Rz rotations to move bloch vector from 0 to "imaginary"
            # qubit
            # (imagine a qubit state signified by the amplitudes at index 2*i
            # and 2*(i+1), corresponding to the select qubits of the
            # multiplexor being in state |i>)
            
            amp_2i = qubit_state_vector[2*state_ind] # amp at 2i
            amp_2i_2= qubit_state_vector[(2*state_ind)+1] #  amp at 2(i+1)
            remaining_qubit_state_vector, theta, phi = Single_qubit_rotation(amp_2i, amp_2i_2)
            
            remaining_vector.append(remaining_qubit_state_vector)
            theta_list.append(-1*theta)
            phi_list.append(-1*phi)
    
    return remaining_vector, theta_list, phi_list

#### old (not quite working)
# def recursive_multiplex(target_gate, list_of_angles, start_qubit_num, end_qubit_num, last_cnot=True):
#     """
#     Args:
#         target_gate (Gate): Ry or Rz gate to apply to target qubit,
#                             multiplexed over all other "select" qubits
                            
#         list_of_angles (list[float]): list of rotation angles to apply Ry and Rz
        
#         last_cnot (bool): add the last cnot if last_cnot = True
#     """
#     number_angles = len(list_of_angles)
#     local_num_qubits = int(np.log2(number_angles)) + 1 # +1 for n+1 qubits!
    
#     qubits_list = cirq.LineQubit.range(start_qubit_num, end_qubit_num)
    
#     LSB = qubits_list[0] # least significant bit
#     MSB = qubits_list[local_num_qubits-1] # most significant bit
    
#     circuit = cirq.Circuit()
    
#     # case of no multiplexing: base case for recursion
#     if local_num_qubits == 1:
#         if target_gate == 'Ry':
#             Ry_gate = cirq.ry(list_of_angles[0])
#             circuit.append(Ry_gate.on(LSB))
#         elif target_gate == 'Rz':
#             Rz_gate = cirq.rz(list_of_angles[0])
#             circuit.append(Rz_gate.on(LSB))
#         else:
#             raise ValueError(f'Incorrect gate specificed: {target_gate}')
        
#         return circuit
    
#     angle_weight = np.kron([[0.5, 0.5], [0.5, -0.5]],
#                                np.identity(2 ** (local_num_qubits - 2)))
    
#     # calc the combo angles
#     list_of_angles = angle_weight.dot(np.array(list_of_angles)).tolist()
    
#     # recursive step on half the angles fulfilling the above assumption
#     multiplex_1 = recursive_multiplex(target_gate, list_of_angles[0:(number_angles // 2)],
#                                       start_qubit_num,
#                                       end_qubit_num-1,
#                                       False)
#     circuit = cirq.Circuit(
#        [
#            circuit.all_operations(),
#            *multiplex_1.all_operations(),
#        ]
#     )
    
#     circuit.append(cirq.CNOT(MSB, LSB))

#     # implement extra efficiency from the paper of cancelling adjacent
#     # CNOTs (by leaving out last CNOT and reversing (NOT inverting) the
#     # second lower-level multiplex)
#     multiplex_2 = recursive_multiplex(target_gate, list_of_angles[(number_angles // 2):],
#                                       start_qubit_num,
#                                       end_qubit_num-1,
#                                       False)
    
#     if number_angles > 1:
#         circuit = cirq.Circuit(
#                                [
#                                    circuit.all_operations(),
#                                    *list(multiplex_2.all_operations())[::-1], # reversed (allowed as circuit is symmetric)
#                                ]
#                             )
#     else:
#         circuit = cirq.Circuit(
#                                [
#                                    circuit.all_operations(),
#                                    *multiplex_2.all_operations(),
#                                ]
#                             )
#     # attach a final CNOT
#     if last_cnot:
#         circuit.append(cirq.CNOT(MSB, LSB))
    
#     return circuit

# def disentangle_circuit(qubit_state_vector, start_qubit_ind, end_qubit_ind):
#     """
#     """

#     circuit = cirq.Circuit()
#     n_qubits = np.log2(len(qubit_state_vector))
#     end_ind_corr = end_qubit_ind

#     if n_qubits!= len(list(range(start_qubit_ind, end_ind_corr))):
#         raise ValueError('incorrect qubit defined qubit indices!')

#     remaining_vector = deepcopy(qubit_state_vector)
#     for qubit_ind in range(start_qubit_ind, end_qubit_ind+1):
#         # work out which rotations must be done to disentangle the LSB
#         # qubit (we peel away one qubit at a time)
#         remaining_vector, theta_list, phi_list = Rotations_to_disentangle(remaining_vector)
        
        
#         add_last_cnot = True
#         if np.linalg.norm(phi_list) != 0 and np.linalg.norm(theta_list) != 0:
#             add_last_cnot = False

#         if np.linalg.norm(phi_list) != 0:
#             rz_mult_circuit = recursive_multiplex('Rz',
#                                                   phi_list,
#                                                   qubit_ind,
#                                                   start_qubit_ind+end_ind_corr,
#                                                   last_cnot=add_last_cnot)
#             circuit.append(rz_mult_circuit)

#         if np.linalg.norm(theta_list) != 0:
#             ry_mult_circuit = recursive_multiplex('Ry',
#                                                   theta_list,
#                                                   qubit_ind,
#                                                   start_qubit_ind+end_ind_corr,
#                                                   last_cnot=add_last_cnot)
#             circuit = cirq.Circuit(
#                        [
#                            circuit.all_operations(),
#                            *list(ry_mult_circuit.all_operations())[::-1],
#                        ]
#                     )
#     return circuit

# def intialization_circuit(qubit_state_vector, start_qubit_ind, check_circuit=False, threshold=7):
#     """
#     Function to create arbitrary state.

#     TODO: currently have very hacky implementation! qubit ordering unusual, requring either to specify order in simulation 
#           (see qubit_order in simulate - commented out)
#           OR reversing each bit and defining new state vector (e.g. 100 becomes 001). Should update this, but does work!

#     """

#     n_qubits = np.log2(len(qubit_state_vector))
    
#     if np.ceil(n_qubits) != np.floor(n_qubits):
#         raise ValueError('state vector is not a qubit state')
    
#     n_qubits = int(n_qubits)

#     if not np.isclose(sum(np.abs(qubit_state_vector)**2), 1):
#         raise ValueError('state vector is not normalized')
    
#     ### hacky! ###
#     re_ordered_state = np.zeros(2**n_qubits, dtype=complex)
#     for ind, amp in enumerate(qubit_state_vector):
#         b_str = Get_state_as_str(n_qubits, ind)
#         new_ind = int(b_str[::-1],2)
#         re_ordered_state[new_ind]=amp

#     old_vec = deepcopy(qubit_state_vector)
#     qubit_state_vector = re_ordered_state.tolist()
#     ####

#     disentangling_circuit = disentangle_circuit(qubit_state_vector, start_qubit_ind, n_qubits+start_qubit_ind)
#     inverse_circuit = cirq.inverse(disentangling_circuit)
    
#     if check_circuit is True:
#         s=cirq.Simulator()
#         results=s.simulate(inverse_circuit)
#         # results=s.simulate(inverse_circuit, qubit_order=list(cirq.LineQubit.range(end_qubit_ind,start_qubit_ind-1,-1)))

#         if not np.allclose(np.around(np.array(old_vec, dtype='complex64'), threshold),np.around(results.state_vector(), threshold)):
#             raise ValueError('circuit not preparing correct state!')
#     return inverse_circuit

def R_full_decomp(target_gate, Angle, control_list, start_qubit_ind, include_last_CNOT=True, last_control_bit=None):
    
    """
    Function deomposes multicontrol Rz or Ry gate into single control rotations and CNOT gates.

    Args:
        target_gate (Str): Target gate: 'Rz' or 'Ry'
        Angle (float): angle to rotate
        control_list (list): list of bits (int) that define control line states
        include_last_CNOT (bool): Whether to include last CNOT
        last_control_bit: Required to keep track of angle sign (DO NOT CHANGE THIS)


    0: ───@───────
          │
    1: ───@───────
          │
    2: ───@───────
          │
    3: ───Ry(π)───
    
    becomes

    0: ─────────────────────────────────────────────────────────────────────@─────────────────────────────────────────────────────────────────────@───
                                                                            │                                                                     │
    1: ──────────────────────────────────@──────────────────────────────────┼──────────────────────────────────@──────────────────────────────────┼───
                                         │                                  │                                  │                                  │
    2: ────────────────@─────────────────┼────────────────@─────────────────┼────────────────@─────────────────┼────────────────@─────────────────┼───
                       │                 │                │                 │                │                 │                │                 │
    3: ───Ry(0.125π)───X───Ry(-0.125π)───X───Ry(0.125π)───X───Ry(-0.125π)───X───Ry(0.125π)───X───Ry(-0.125π)───X───Ry(0.125π)───X───Ry(-0.125π)───X───

    
    """
    
    line_qubit_list = cirq.LineQubit.range(start_qubit_ind, start_qubit_ind+len(control_list)+1)
    
    control_list = control_list[::-1]
    circuit= cirq.Circuit()
    LSB = line_qubit_list[-1]
    # case of no multiplexing: base case for recursion
    if len(line_qubit_list)==1:
        if target_gate == 'Ry':
            Ry_gate = cirq.ry(Angle)
            circuit.append(Ry_gate.on(LSB))
        elif target_gate == 'Rz':
                Rz_gate = cirq.rz(Angle)
                circuit.append(Rz_gate.on(LSB))
        else:
            raise ValueError(f'Incorrect gate specificed: {target_gate}')
    
        return circuit
    
    
    if ((target_gate == 'Rz') and last_control_bit==1):
        Angle_left = Angle/2
        Angle_right = -Angle/2  # note sign here!
    else:
        Angle_left = Angle/2
        Angle_right = Angle/2  

    MSB = line_qubit_list[0]
    last_control_bit = control_list[-1]
    
    Angle_left = Angle/2
    decomp_left = R_full_decomp(target_gate, 
                                Angle_left, 
                                control_list[:-1], 
                                start_qubit_ind+1, 
                                last_control_bit=last_control_bit,
                                include_last_CNOT=False)

    circuit = cirq.Circuit(
       [
           circuit.all_operations(),
           *decomp_left.all_operations(),
       ]
    )
        
    circuit.append(cirq.CNOT(MSB, LSB))
    
    if ((target_gate == 'Rz') and (last_control_bit==1)) or ((target_gate == 'Ry') and (last_control_bit==1)):
        Angle_right = -Angle/2  # note sign here!
    else:
        Angle_right = Angle/2  # note sign here!
    
    decomp_right = R_full_decomp(target_gate, 
                    Angle_right, 
                        control_list[:-1], 
                        start_qubit_ind+1, 
                    last_control_bit=last_control_bit,
                    include_last_CNOT=False)
    
    if len(control_list) > 1:
       
        circuit = cirq.Circuit(
                               [
                                   circuit.all_operations(),
                                   *list(decomp_right.all_operations())[::-1], # reversed (allowed as circuit is symmetric)
                               ]
                            )
    else:
            
        circuit = cirq.Circuit(
                               [
                                   circuit.all_operations(),
                                   *decomp_right.all_operations(),
                               ]
                            )
    # attach a final CNOT
    if include_last_CNOT:
        circuit.append(cirq.CNOT(MSB, LSB))
    return circuit

def R_angle_list(target_gate, list_of_angles, start_qubit_ind=0):
    """
    Function decomposes list of angles and target gate type into composite single and two qubit gates, using R_full_decomp function.

    Args:
        target_gate (str) = 'Rz' or 'Ry'
        list_of_angles (list) = list of angles (float)
        start_qubit_ind(int) = start qubit index

    ** Example ** 
    
    theta_list = [-1.5707963267948966,-1.5707963267948966, -1.5707963267948966, -1.5707963267948966]
    circuit = R_angle_list('Rz', theta_list, 0)
    print(circuit)

    >>
    0: ───────────────────────────────────@───────────────────────────────────@──────────────────────────────────@──────────────────────────────────@───────────────────────────────────@─────────────────────────────────@──────────────────────────────────@──────────────────────────────────@───
                                          │                                   │                                  │                                  │                                   │                                 │                                  │                                  │
    1: ─────────────────@─────────────────┼─────────────────@─────────────────┼─────────────────@────────────────┼────────────────@─────────────────┼─────────────────@─────────────────┼────────────────@────────────────┼─────────────────@────────────────┼─────────────────@────────────────┼───
                        │                 │                 │                 │                 │                │                │                 │                 │                 │                │                │                 │                │                 │                │
    2: ───Rz(-0.125π)───X───Rz(-0.125π)───X───Rz(-0.125π)───X───Rz(-0.125π)───X───Rz(-0.125π)───X───Rz(0.125π)───X───Rz(0.125π)───X───Rz(-0.125π)───X───Rz(-0.125π)───X───Rz(-0.125π)───X───Rz(0.125π)───X───Rz(0.125π)───X───Rz(-0.125π)───X───Rz(0.125π)───X───Rz(-0.125π)───X───Rz(0.125π)───X───

    """
    
    N_q = int(np.log2(len(list_of_angles)))
    circuit = cirq.Circuit()
    
    for control_ind, angle in enumerate(list_of_angles):
        if angle == 0:
            continue
        
        if N_q ==0:
            control_list=[]
        else:
            control_list = list(map(lambda x: int(x), list(np.binary_repr(control_ind, width=N_q))))
        
        decomp_R = R_full_decomp(target_gate, 
                                 angle,
                                 control_list, 
                                 start_qubit_ind,
                                 include_last_CNOT=True,
                                 last_control_bit=None)
        circuit.append(decomp_R)
    return circuit


def disentangle_circuit(qubit_state_vector, start_qubit_ind=0):
    """
    Generates disentangling quantum circuit according to https://arxiv.org/pdf/quant-ph/0406176v5.pdf

    Args:
        qubit_state_vector (list): list of state amplitudes (complex)
        start_qubit_ind (ind)
    """

    circuit = cirq.Circuit()
    n_qubits = np.log2(len(qubit_state_vector))
    
    if np.ceil(n_qubits) != np.floor(n_qubits):
        raise ValueError('state vector is not a qubit state')
    
    n_qubits = int(n_qubits)
#     print(n_qubits)
    
    N_qubits_remaining_vector= n_qubits
    remaining_vector = deepcopy(qubit_state_vector)
    for qubit_ind in range(start_qubit_ind, n_qubits+start_qubit_ind):
        # work out which rotations must be done to disentangle the LSB
        # qubit (we peel away one qubit at a time)
        remaining_vector, theta_list, phi_list = Rotations_to_disentangle(remaining_vector)
        
        decomp_R = R_angle_list('Rz', phi_list, start_qubit_ind)
        circuit.append(decomp_R)

        decomp_R = R_angle_list('Ry', theta_list, start_qubit_ind)
        circuit.append(decomp_R)
    
    final_global_phase = remaining_vector
    return circuit, final_global_phase

def intialization_circuit(qubit_state_vector, start_qubit_ind, check_circuit=False):
    """
    Function to create arbitrary state. Inverse of disentangling circuit!

    """
    if isinstance(qubit_state_vector, list):
        qubit_state_vector = np.asarray(qubit_state_vector, dtype=complex)
    else:
        qubit_state_vector = np.asarray(qubit_state_vector.flat, dtype=complex)

    n_qubits = np.log2(len(qubit_state_vector))
    
    if np.ceil(n_qubits) != np.floor(n_qubits):
        raise ValueError('state vector is not a qubit state')
    

    if not np.isclose(sum(np.abs(qubit_state_vector)**2), 1):
        raise ValueError('state vector is not normalized')
    

    disentangling_circuit, Global_phase = disentangle_circuit(qubit_state_vector, start_qubit_ind=start_qubit_ind)
    inverse_circuit = cirq.inverse(disentangling_circuit)
    
    if check_circuit is True:
        output_state = inverse_circuit.final_state_vector()
        # results=s.simulate(inverse_circuit, qubit_order=list(cirq.LineQubit.range(end_qubit_ind,start_qubit_ind-1,-1)))

        if not np.allclose(output_state*Global_phase, qubit_state_vector):
            raise ValueError('circuit not preparing correct state!')
    return inverse_circuit, Global_phase


### alternates ###

def project_V_on_U(u_vec, v_vec):
    ### note NOT normalised

    # u_next = (np.dot(u_vec, v_vec)/np.dot(u_vec, u_vec))*u_vec
    u_next = (np.vdot(u_vec, v_vec)/np.vdot(u_vec, u_vec))*u_vec # vdot deals with complex vectors!
    return u_next
    

def Gram_Schmidt(first_col, zero_tolerance=1e-10):
    """
    Given the first column of a unitary matrix, will fill all other columns with ortho vectors via
    Gram-Schmidt process
    
    """
    first_col = np.asarray(first_col, dtype=complex)

    if not np.isclose(sum(np.abs(first_col)**2), 1):
        raise ValueError('first column (input state) is not normalized')
    
    
    # find first non zero amplitude index
    first_non_zero_ind = None
    for ind, amp in enumerate(first_col.flat):
        if np.isclose(np.abs(amp),0, atol=zero_tolerance):
            continue
        else:
            first_non_zero_ind = ind
            break

    new_matrix = np.eye(len(first_col.flat), dtype=complex)
    # make sure linearly independent, by putting state along column with first non zero index
    if first_non_zero_ind==0:
        new_matrix[:,0] = first_col/np.linalg.norm(np.abs(first_col), ord=2)
    else:
        new_matrix[:,first_non_zero_ind] = first_col/np.linalg.norm(np.abs(first_col), ord=2)
        new_matrix[:,[0, first_non_zero_ind]] = new_matrix[:,[first_non_zero_ind, 0]]

    for i in range(1,len(first_col.flat)):
        vec_V = new_matrix[:,i]
        ortho = sum([project_V_on_U(new_matrix[:,ortho_col_ind], vec_V) for ortho_col_ind in range(0,i)])
        new_vec_V = vec_V - ortho
        norm_vec_V = new_vec_V/np.linalg.norm(np.abs(new_vec_V),ord=2)
        new_matrix[:,i] =norm_vec_V
            
    
    return new_matrix

### IBM function
from qiskit.extensions import UnitaryGate
from qiskit import QuantumCircuit, Aer, execute
from qiskit.compiler import transpile
from cirq.contrib.qasm_import import circuit_from_qasm
def prepare_arb_state_IBM_to_cirq(state_vector, start_qubit_ind=0, opt_level=0,allowed_gates=['id', 'rz', 'ry', 'rx', 'cx' ,'s', 'h', 'y','z', 'x']):
    # TODO: bug when using IBM's transpiler
    raise ValueError('Function not working properly')

    UnitaryMatrix = Gram_Schmidt(state_vector)

    qiskit_matrix_gate = UnitaryGate(UnitaryMatrix)

    n_qubits = int(np.log2(UnitaryMatrix.shape[0]))

    qiskit_c = QuantumCircuit(n_qubits)
    qiskit_c.unitary(qiskit_matrix_gate, list(range(0,n_qubits)), label='initialize')

    compiled_circuit = transpile(qiskit_c,
                                optimization_level=opt_level, 
                                basis_gates=allowed_gates, 
                                approximation_degree=1)

    ibm_qasm = compiled_circuit.qasm()

    cirq_circuit = circuit_from_qasm(ibm_qasm)


    ### check global phase
    mat1 = cirq_circuit.unitary()[:,0]
    mat2 = np.asarray(state_vector, dtype=complex)
    for ind, elt in enumerate(mat1.flat):
            if abs(elt) > 0:
                original_term = elt
                new_term = mat2.flat[ind]

                global_phase = np.angle(original_term/new_term) # find phase difference between unitaries!

    if not np.isclose(global_phase, 0):
        qubit = list(cirq_circuit.all_qubits())[0]
        op1 = cirq.ZPowGate(exponent=2, global_shift=global_phase/(2*np.pi)).on(qubit) # exponent 2 hence divided global phase by 2 (note also divide by pi as in units of pi in ZpowGate definition)
        cirq_circuit.append(op1)


    ## rename qubits as line qubits and add start qubit index if necessary
    sorted_original_qubits = [cirq.LineQubit(i) for i in range(start_qubit_ind, start_qubit_ind+n_qubits)]
    sorted_named_qubits = sorted(list(cirq_circuit.all_qubits()), key= lambda NamedQ: int(NamedQ.name[2:]))

    NamedQ_to_LineQ_dict = dict(zip(sorted_named_qubits, sorted_original_qubits))
    cirq_circuit = cirq_circuit.transform_qubits(lambda x: NamedQ_to_LineQ_dict[x])

    return cirq_circuit



## cirq matrix gate method
def prepare_arb_state_cirq_matrix_gate(state_vector, start_qubit_ind=0):
    
    if not np.isclose(sum(np.abs(state_vector)**2), 1):
        raise ValueError('state_vector is not valid quantum state (not normalized)')

    UnitaryMatrix = Gram_Schmidt(state_vector)

    n_qubits = int(np.log2(UnitaryMatrix.shape[0]))

    qubits = list(cirq.LineQubit.range(start_qubit_ind, start_qubit_ind+n_qubits))
    state_prep_circuit = cirq.Circuit(cirq.MatrixGate(UnitaryMatrix).on(*qubits))

    return state_prep_circuit