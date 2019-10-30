

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

