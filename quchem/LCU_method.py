# https://arxiv.org/pdf/quant-ph/0104030.pdf
# ^^^ Need to be able to prepare arbitrary state!
import numpy as np


# state = |000> + |001> + |010> + |011> + |100> + |101 > + |110 > + |111>
constants = [0.5, 0.25, 0.3, 0.1, 0.1, 0.14, 0.2, 0.71965]
num_qubits = 3

def Get_state_as_str(n_qubits, integer):
    bin_str_len = '{' + "0:0{}b".format(n_qubits) + '}'
    return bin_str_len.format(integer)

#state_list = [Get_state_as_str(num_qubits, i) for i in range(2**num_qubits)]

states = np.eye(2**num_qubits, dtype=int)
x = np.array2string(states, separator='')
state_list = [x[12*i+2: 12*i+2+8] for i in range(2**num_qubits)]

alpha_j={}
for j in np.arange(2,len(constants)-1, 1): # for j=2 to j=n-1
    # for i in np.arange(j,len(constants),1):
    upper_terms = [Get_state_as_str(j-1, i) + "1" for i in np.arange(0,j,1)]
    lower_terms = [Get_state_as_str(j-1, i) + "0" for i in np.arange(0, j, 1)]

    for k in range(len(upper_terms)):
        upper_str = upper_terms[k]
        lower_str = lower_terms[k]

        upper_sum=[]
        lower_sum=[]
        for i in range(len(state_list)):
            if state_list[i][0:j] == upper_str:
                upper_sum.append(constants[i]**2)
            if state_list[i][0:j] == lower_str:
                lower_sum.append(constants[i] ** 2)
        print(j,k)
        alpha_j[(j,k)] = np.arctan(sum(upper_sum)/sum(lower_sum))

alpha_j={}
for j in np.arange(1,len(state_list)-1,1): # for j=2 to j=n-1
    remaining_terms = [state_list[i][j:] for i in range(len(state_list))]
    print(remaining_terms)

