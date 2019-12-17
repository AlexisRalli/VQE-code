import numpy as np
from functools import reduce

anti_commuting_sets = \
{0: [('I0 I1 I2 I3', (-0.09706626861762581+0j))],
 1: [('Z0 Z1 I2 I3', (0.168688981686933+0j))],
 2: [('Z0 I1 Z2 I3', (0.12062523481381841+0j))],
 3: [('Z0 I1 I2 Z3', (0.16592785032250779+0j))],
 4: [('I0 Z1 Z2 I3', (0.16592785032250779+0j))],
 5: [('I0 Z1 I2 Z3', (0.12062523481381841+0j))],
 6: [('I0 I1 Z2 Z3', (0.1744128761065161+0j))],
 7: [('Z0 I1 I2 I3', (0.17141282639402383+0j)),
  ('Y0 X1 X2 Y3', (0.045302615508689394+0j))],
 8: [('I0 Z1 I2 I3', (0.1714128263940239+0j)),
  ('Y0 Y1 X2 X3', (-0.045302615508689394+0j))],
 9: [('I0 I1 Z2 I3', (-0.22343153674663985+0j)),
  ('X0 X1 Y2 Y3', (-0.045302615508689394+0j))],
 10: [('I0 I1 I2 Z3', (-0.22343153674663985+0j)),
  ('X0 Y1 Y2 X3', (0.045302615508689394+0j))]}


def Commute(P1, P2):
    P1 = P1.split(' ')
    P2 = P2.split(' ')

    checker = np.zeros(len(P1))
    for i in range(len(P1)):
        if P1[i][0] == P2[i][0]:
            checker[i] = 1
        elif P1[i][0] == 'I' or P2[i][0] == 'I':
            checker[i] = 1
        else:
            checker[i] = -1

    if reduce((lambda x, y: x * y), checker) == 1:
        return True
    else:
        return False

print(Commute(anti_commuting_sets[7][0][0], anti_commuting_sets[7][1][0]))

for key in anti_commuting_sets:
    selected_set = anti_commuting_sets[key]

    for P_top in selected_set:
        PauliWord=P_top[0]


