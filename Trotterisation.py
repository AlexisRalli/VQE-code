

QWC_PauliWords = {
    0: [('Y0 X1 X2 Y3', (0.04919764587885283+0j)),
        ('I0 I1 I2 I3', (-0.32760818995565577+0j))],
    1: [('Y0 Y1 X2 X3', (-0.04919764587885283+0j))],
    2: [('X0 X1 Y2 Y3', (-0.04919764587885283+0j))],
    3: [('X0 Y1 Y2 X3', (0.04919764587885283+0j))],
    4: [('Z0 I1 I2 I3', (0.1371657293179602+0j)),
        ('I0 Z1 I2 I3', (0.1371657293179602+0j)),
        ('I0 I1 Z2 I3', (-0.13036292044009176+0j)),
        ('I0 I1 I2 Z3', (-0.13036292044009176+0j)),
        ('Z0 Z1 I2 I3', (0.15660062486143395+0j)),
        ('Z0 I1 Z2 I3', (0.10622904488350779+0j)),
        ('Z0 I1 I2 Z3', (0.15542669076236065+0j)),
        ('I0 Z1 Z2 I3', (0.15542669076236065+0j)),
        ('I0 Z1 I2 Z3', (0.10622904488350779+0j)),
        ('I0 I1 Z2 Z3', (0.1632676867167479+0j))]
                }

for key, value in QWC_PauliWords.items():
    for PauliWord, constant in value:
        print(PauliWord)



# note that sum of constants squared = 1 !!!!
anti_set = [('Y0 X1 X2 Y3', (0.25)),
           ('Y0 Y1 X2 X3', (0.25)),
           ('X0 X1 Y2 Y3', (0.25)),
           ('X0 Y1 Y2 X3', (0.25))]


import numpy as np
X_sk = np.zeros((len(anti_set), len(anti_set)), dtype = object)

for s in range(len(anti_set)):
    P_s = anti_set[s][0]
    for k in range(len(anti_set)):
        P_k = anti_set[k][0]
        X_sk[s][k] = (P_s, P_k)



beta_list = [const for PauliWord, const in anti_set]
theta_sk = np.zeros((len(anti_set), len(anti_set)))

# eqn 16!!!!!!
for s in range(len(anti_set)):
    beta_s = beta_list[s]

    for k in range(len(anti_set)):
        beta_k = beta_list[k]

        sum = 0
        for j in range(k-1):
            sum += beta_list[j]**2

        theta_sk[s][k] = np.arctan( beta_k / (beta_s**2 + sum) )


### can know build operator (eqn 12):

Operator_X_sk = np.zeros((len(anti_set), len(anti_set)), dtype = object)
for s in range(len(anti_set)):
    for k in range(len(anti_set)):
        Operator_X_sk[s][k] = (*X_sk[s][k], theta_sk[s][k])



from scipy.sparse import bsr_matrix
X = bsr_matrix(np.array([[0, 1],
                         [1, 0]]))
Y = bsr_matrix(np.array([[0, -1j],
                         [1j, 0]]))
Z = bsr_matrix(np.array([[1, 0],
                         [0, -1]]))
I = bsr_matrix(np.array([[1, 0],
                         [0, 1]]))


for s in range(len(anti_set)):
    for k in range(len(anti_set)):
        for obj in Operator_X_sk[s][k]:
            if isinstance(obj, str):

                PauliWord = obj.split(sep = ' ')


                for pauli in PauliWord:



string = 'Y0 X1 X2 Y3'
string.split(sep = ' ')




OperatorsKeys = {
    'X': X,
    'Y': Y,
    'Z': Z,
    'I': I,
}


def X_sk_Operators():
    '''

    Self-inverse operators:
    X_sk = i * P_s * P_k

    where 0 =< k =< s-1
    # note we are only varying over k!

    note when:
    j!=s and j != k
    the X_sk will COMMUTE

    otherwise it ANTI-COMMUTES with P_k and P_s

    note number of permutations:
    n!/(n-r)!
    [n = number of terms in set and r = number of indices, in this case 2 (s and k)]
    e.g. in above case have
    s=0, k = 0,1,2,3,4
    s=1, k = 0,1,2,3,4
    ...
    s=4, k = 0,1,2,3,4

    aka 5!/(5-2)! = 20 terms!

    :return:
    '''

