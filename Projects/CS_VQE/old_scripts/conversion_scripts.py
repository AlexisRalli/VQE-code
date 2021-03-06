from openfermion.ops import QubitOperator

def convert_op_str(qubit_op_str, op_coeff):
    """
    Convert qubit operator into openfermion format
    
    """
    converted_Op=[f'{qOp_str}{qNo_index}' for qNo_index, qOp_str in enumerate(qubit_op_str) if qOp_str !='I']
    
    seperator = ' ' #space
    Openfermion_qubit_op = QubitOperator(seperator.join(converted_Op), op_coeff)
    
    return Openfermion_qubit_op

def Get_Openfermion_Hamiltonian(hamiltonian_dict):
    """
    given a dictionary of {qubit_op_str: coeff}
    
    return Openfermion QubitOperator of Hamiltonian
    
    """
    qubit_Hamiltonian=QubitOperator()
    
    for qubit_op_str, op_coeff in hamiltonian_dict.items():
        
        openFermion_q_op = convert_op_str(qubit_op_str, op_coeff)
        qubit_Hamiltonian+=openFermion_q_op
        
    return qubit_Hamiltonian


def Openfermion_to_dict(QubitOp, N_qubits):

    op_dict={}
    for op in QubitOp:
        PauliStr, coeff = tuple(*op.terms.items())
        
        if PauliStr:
            qNo_list, qPstr_list = zip(*PauliStr)
        else:
            # identity operator
            I_op = 'I'*N_qubits
            op_dict[I_op] = coeff
            continue


        P_str_key = ''
        running_index=0
        for i in range(N_qubits):
            if i in qNo_list:
                P_str_key+=f'{qPstr_list[running_index]}'
                running_index+=1
            else:
                P_str_key+='I'
        op_dict[P_str_key] = coeff

    return op_dict
