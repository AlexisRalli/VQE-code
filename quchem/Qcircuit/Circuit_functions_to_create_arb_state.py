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
#     zero_state_amp = complex(zero_state_amp)
#     one_state_amp = complex(one_state_amp)
    
#     norm = np.sqrt(zero_state_amp**2 + one_state_amp**2 )
    
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
    Get rotation to create passed in qubit state
    
    |psi> = cos(theta/2) |0> + e^{-1j * phi/2} sin(theta/2) |1>
    
    See equation 8 and 9 of https://arxiv.org/pdf/quant-ph/0406176v5.pdf
    
    """
    # zero_state_amp = complex(zero_state_amp)
    # one_state_amp = complex(one_state_amp)
    zero_state_amp =np.complex128(zero_state_amp)
    one_state_amp = np.complex128(one_state_amp)
    
    norm = np.sqrt(zero_state_amp**2 + one_state_amp**2 )
    
    if np.isclose(norm, 0):
        theta = 0
        a_arg = 0
        b_arg = 0
        final_t = 0
        phi = 0
    else:
        # theta= 2 * np.arccos(np.abs(zero_state_amp) / norm)
        theta=2*np.arctan2(np.abs(one_state_amp)/ norm.real, np.abs(zero_state_amp)/ norm.real)

        a_arg = np.angle(zero_state_amp)
        b_arg = np.angle(one_state_amp)
        final_t = a_arg + b_arg
        phi = b_arg - a_arg
    
    return norm * np.exp(1.J * final_t / 2), theta, phi

def Rotations_to_disentangle(qubit_state_vector):
    """
    pg 11 of https://arxiv.org/pdf/quant-ph/0406176v5.pdf
    
    Method to work out Ry and Rz rotation angles to disentangle the least significant bit (LSB).
    These rotations make up a block diagonal matrix U == multiplexor
    
    
    
    ### futher background:
    
    Given |ψ> an (n+1) qubit state, seperate the state into a separable (unentanged) state by the following circuit:
    
               n  :──\\───(C)───────────────(C)────────────
   |ψ>                     │                 │             
               n+1:─────── Rz (-phi) ─────── Ry(-theta)──── |ψ''>


    the 2^{n+1} state vector is split into TWO 2^{n} states... This can be done by the circuit above.
    
    Overall |ψ> (2^{n+1} state vector) is split into 2^{n} contiguous 2-element blocks. This can be interpreted as a
    2D complex vector. We can label this |ψ_{c}>
    
    Then:
    
    Rz(-φ_{c}) Ry(-θ_{c}) |ψ> = r_{c} exp(1j*t_{c}) |0 >
    
    |ψ''> is the n-qubit state given by the 2^{n}-element row vector with c-th entry r_{c} exp(1j*t_{c}).
    
    If we let U be the block diagonal sum ⊕_{c} Ry(-θ_{c}) Rz(-φ_{c}). THEN:
    
                U |ψ> = |ψ''> |0>
    
    We can implement U as a multiplexed Rz gate followed by a multiplexed Ry gate!
    
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

def recursive_multiplex(target_gate, list_of_angles, start_qubit_num, end_qubit_num, last_cnot=True):
    """
    Args:
        target_gate (Gate): Ry or Rz gate to apply to target qubit,
                            multiplexed over all other "select" qubits
                            
        list_of_angles (list[float]): list of rotation angles to apply Ry and Rz
        
        last_cnot (bool): add the last cnot if last_cnot = True
    """
    number_angles = len(list_of_angles)
    local_num_qubits = int(np.log2(number_angles)) + 1 # +1 for n+1 qubits!
    
    qubits_list = cirq.LineQubit.range(start_qubit_num, end_qubit_num)
    
    LSB = qubits_list[0] # least significant bit
    MSB = qubits_list[local_num_qubits-1] # most significant bit
    
    circuit = cirq.Circuit()
    
    # case of no multiplexing: base case for recursion
    if local_num_qubits == 1:
        if target_gate == 'Ry':
            Ry_gate = cirq.ry(list_of_angles[0])
            circuit.append(Ry_gate.on(LSB))
        elif target_gate == 'Rz':
            Rz_gate = cirq.rz(list_of_angles[0])
            circuit.append(Rz_gate.on(LSB))
        else:
            raise ValueError(f'Incorrect gate specificed: {target_gate}')
        
        return circuit
    
    angle_weight = np.kron([[0.5, 0.5], [0.5, -0.5]],
                               np.identity(2 ** (local_num_qubits - 2)))
    
    # calc the combo angles
    list_of_angles = angle_weight.dot(np.array(list_of_angles)).tolist()
    
    # recursive step on half the angles fulfilling the above assumption
    multiplex_1 = recursive_multiplex(target_gate, list_of_angles[0:(number_angles // 2)],
                                      start_qubit_num,
                                      end_qubit_num-1,
                                      False)
    circuit = cirq.Circuit(
       [
           circuit.all_operations(),
           *multiplex_1.all_operations(),
       ]
    )
    
    circuit.append(cirq.CNOT(MSB, LSB))

    # implement extra efficiency from the paper of cancelling adjacent
    # CNOTs (by leaving out last CNOT and reversing (NOT inverting) the
    # second lower-level multiplex)
    multiplex_2 = recursive_multiplex(target_gate, list_of_angles[(number_angles // 2):],
                                      start_qubit_num,
                                      end_qubit_num-1,
                                      False)
    
    if number_angles > 1:
        circuit = cirq.Circuit(
                               [
                                   circuit.all_operations(),
                                   *list(multiplex_2.all_operations())[::-1],
                               ]
                            )
    else:
        circuit = cirq.Circuit(
                               [
                                   circuit.all_operations(),
                                   *multiplex_2.all_operations(),
                               ]
                            )
    # attach a final CNOT
    if last_cnot:
        circuit.append(cirq.CNOT(MSB, LSB))
    
    return circuit

def disentangle_circuit(qubit_state_vector, start_qubit_ind, end_qubit_ind):
    """
    """

    circuit = cirq.Circuit()
    n_qubits = np.log2(len(qubit_state_vector))
    end_ind_corr = end_qubit_ind+1 # as index from 0

    if n_qubits!= len(list(range(start_qubit_ind, end_ind_corr))):
        raise ValueError('incorrect qubit defined qubit indices!')

    remaining_vector = deepcopy(qubit_state_vector)
    for qubit_ind in range(start_qubit_ind, end_qubit_ind+1):
        # work out which rotations must be done to disentangle the LSB
        # qubit (we peel away one qubit at a time)
        remaining_vector, theta_list, phi_list = Rotations_to_disentangle(remaining_vector)
        
        
        add_last_cnot = True
        if np.linalg.norm(phi_list) != 0 and np.linalg.norm(theta_list) != 0:
            add_last_cnot = False

        if np.linalg.norm(phi_list) != 0:
            rz_mult_circuit = recursive_multiplex('Rz',
                                                  phi_list,
                                                  qubit_ind,
                                                  start_qubit_ind+end_ind_corr,
                                                  last_cnot=add_last_cnot)
            circuit.append(rz_mult_circuit)

        if np.linalg.norm(theta_list) != 0:
            ry_mult_circuit = recursive_multiplex('Ry',
                                                  theta_list,
                                                  qubit_ind,
                                                  start_qubit_ind+end_ind_corr,
                                                  last_cnot=add_last_cnot)
            circuit = cirq.Circuit(
                       [
                           circuit.all_operations(),
                           *list(ry_mult_circuit.all_operations())[::-1],
                       ]
                    )
    return circuit

def intialization_circuit(qubit_state_vector, start_qubit_ind, end_qubit_ind, check_circuit=False, threshold=7):
    """
    Function to create arbitrary state.

    TODO: currently have very hacky implementation! qubit ordering unusual, requring either to specify order in simulation 
          (see qubit_order in simulate - commented out)
          OR reversing each bit and defining new state vector (e.g. 100 becomes 001). Should update this, but does work!

    """

    n_qubits = np.log2(len(qubit_state_vector))
    end_ind_corr = end_qubit_ind+1
    
    if np.ceil(n_qubits) != np.floor(n_qubits):
        raise ValueError('state vector is not a qubit state')
    
    n_qubits = int(n_qubits)

    if not np.isclose(sum(np.abs(qubit_state_vector)**2), 1):
        raise ValueError('state vector is not normalized')
    
    ### hacky! ###
    re_ordered_state = np.zeros(2**n_qubits)
    for ind, amp in enumerate(qubit_state_vector):
        b_str = Get_state_as_str(n_qubits, ind)
        new_ind = int(b_str[::-1],2)
        re_ordered_state[new_ind]=amp

    old_vec = deepcopy(qubit_state_vector)
    qubit_state_vector = re_ordered_state.tolist()
    ####

    disentangling_circuit = disentangle_circuit(qubit_state_vector, start_qubit_ind, end_qubit_ind)
    inverse_circuit = cirq.inverse(disentangling_circuit)
    
    if check_circuit is True:
        s=cirq.Simulator()
        results=s.simulate(inverse_circuit)
        # results=s.simulate(inverse_circuit, qubit_order=list(cirq.LineQubit.range(end_qubit_ind,start_qubit_ind-1,-1)))

        if not np.allclose(np.around(np.array(old_vec, dtype='complex64'), threshold),np.around(results.state_vector(), threshold)):
            raise ValueError('circuit not preparing correct state!')
    return inverse_circuit