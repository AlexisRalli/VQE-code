

QWC_PauliWords = {0: ['Y0 X1 X2 Y3', 'I0 I1 I2 I3'],
                  1: ['Y0 Y1 X2 X3'],
                  2: ['X0 X1 Y2 Y3'],
                  3: ['X0 Y1 Y2 X3'],
                  4: ['Z0 I1 I2 I3',
                   'I0 Z1 I2 I3',
                   'I0 I1 Z2 I3',
                   'I0 I1 I2 Z3',
                   'Z0 Z1 I2 I3',
                   'Z0 I1 Z2 I3',
                   'Z0 I1 I2 Z3',
                   'I0 Z1 Z2 I3',
                   'I0 Z1 I2 Z3',
                   'I0 I1 Z2 Z3']}

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

