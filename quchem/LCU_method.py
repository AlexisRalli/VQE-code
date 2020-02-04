# https://arxiv.org/pdf/quant-ph/0104030.pdf
# ^^^ Need to be able to prepare arbitrary state!
import numpy as np
import cirq

def Get_state_as_str(n_qubits, integer):
    bin_str_len = '{' + "0:0{}b".format(n_qubits) + '}'
    return bin_str_len.format(integer)

# state = |000> + |001> + |010> + |011> + |100> + |101 > + |110 > + |111>


num_qubits = 3
constants = np.random.rand(2 ** num_qubits)
state_list = [Get_state_as_str(num_qubits, i) for i in range(2 ** num_qubits)]
alpha_j = {}
for j in np.arange(1, num_qubits + 1, 1):  # for j=1 to j=n-1
    if j == 1:
        upper_list_full = set(['1' + state[j:] for state in state_list])
        lower_list_full = set(['0' + state[j:] for state in state_list])

        upper_sum = []
        lower_sum = []
        for i in range(len(state_list)):
            state = state_list[i]
            if state in upper_list_full:
                upper_sum.append(constants[i] ** 2)
            elif state in lower_list_full:
                lower_sum.append(constants[i] ** 2)
        alpha_j[(j, 0)] = np.arctan(np.sqrt(sum(upper_sum) / sum(lower_sum)))

    elif j == num_qubits:
        upper_list_full = list(set([state[:j-1] + '1' for state in state_list]))
        lower_list_full = list(set([state[:j-1] + '0' for state in state_list]))

        for k in range(len(upper_list_full)):
            upper_term = upper_list_full[k]
            lower_term = lower_list_full[k]

            upper_sum = []
            lower_sum = []
            for i in range(len(state_list)):
                state = state_list[i]
                if state == upper_term:
                    upper_sum.append(constants[i])
                elif state == lower_term:
                    lower_sum.append(constants[i])
            print('sum: ', upper_sum)
            alpha_j[(j, k)] = np.arctan(sum(upper_sum) / sum(lower_sum))  # note no sqrt!

    else:
        print("###")
        print(j)
        for ii in np.arange(0, 2 ** (j - 1), 1):
            # print('i_list:', ii)
            upper_str = Get_state_as_str(j - 1, ii) + '1'
            lower_str = Get_state_as_str(j - 1, ii) + '0'

            upper_list_full = set([upper_str + state[j:] for state in state_list])
            lower_list_full = set([lower_str + state[j:] for state in state_list])
            # print(upper_list_full)
            # print(lower_list_full)
            lower_sum = []
            upper_sum = []
            for i in range(len(state_list)):
                state = state_list[i]
                if state in upper_list_full:
                    upper_sum.append(constants[i] ** 2)
                elif state in lower_list_full:
                    lower_sum.append(constants[i] ** 2)
            alpha_j[(j, ii)] = np.arctan(np.sqrt(sum(upper_sum) / sum(lower_sum)))
alpha_j
# NOTE : control is ii value in alpha_j dict!


class My_U_Gate(cirq.SingleQubitGate):
    """
    Description

    Args:
        theta (float): angle to rotate by in radians.
        number_control_qubits (int): number of control qubits
    """

    def __init__(self, theta, number_control_qubits):
        self.theta = theta
        self.num_control_qubits = number_control_qubits
    def _unitary_(self):
        Unitary_Matrix = np.array([
                    [np.cos(self.theta), np.sin(self.theta)],
                    [np.sin(self.theta), -1* np.cos(self.theta)]
                ])
        return Unitary_Matrix
    def num_qubits(self):
        return 1
    # def _circuit_diagram_info_(self, args):
    #     return 'U = {} rad'

    def _circuit_diagram_info_(self,args):
        return cirq.CircuitDiagramInfo(
            wire_symbols=tuple([*['@' for _ in range(self.num_control_qubits-1)],' U = {} rad '.format(self.theta)]),exponent=1)

if __name__ == '__main__':
    theta = np.pi

    U_single_qubit = My_U_Gate(theta,0)
    op = U_single_qubit.on(cirq.LineQubit(1))
    print(cirq.Circuit.from_ops(op))

    U_multi_qubit = My_U_Gate(theta,3)
    op2 = U_multi_qubit.on(cirq.LineQubit(1)).controlled_by(cirq.LineQubit(2),cirq.LineQubit(3),cirq.LineQubit(4))
    print(cirq.Circuit.from_ops(op2))