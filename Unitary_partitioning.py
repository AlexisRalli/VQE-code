import numpy as np

anti_commuting_sets = {
     0: [('I0 I1 I2 I3', (-0.32760818995565577+0j))],
     1: [('Z0 Z1 I2 I3', (0.15660062486143395+0j))],
     2: [('Z0 I1 Z2 I3', (0.10622904488350779+0j))],
     3: [('Z0 I1 I2 Z3', (0.15542669076236065+0j))],
     4: [('I0 Z1 Z2 I3', (0.15542669076236065+0j))],
     5: [('I0 Z1 I2 Z3', (0.10622904488350779+0j))],
     6: [('I0 I1 Z2 Z3', (0.1632676867167479+0j))],
     7: [('Z0 I1 I2 I3', (0.1371657293179602+0j)), ('Y0 X1 X2 Y3', (0.04919764587885283+0j)), ('X0 I1 I2 I3', (0.04919764587885283+0j))], # <- I added this term to check code
     8: [('I0 Z1 I2 I3', (0.1371657293179602+0j)), ('Y0 Y1 X2 X3', (-0.04919764587885283+0j))],
     9: [('I0 I1 Z2 I3', (-0.13036292044009176+0j)),('X0 X1 Y2 Y3', (-0.04919764587885283+0j))],
     10: [('I0 I1 I2 Z3', (-0.13036292044009176+0j)), ('X0 Y1 Y2 X3', (0.04919764587885283+0j))]
}

# for key, value in anti_commuting_sets.items():
#     for PauliWord, constant in value:
#         print(PauliWord)



def Get_beta_j_cofactors(anti_commuting_sets):
    for key, value in anti_commuting_sets.items():
        factor = sum([constant**2 for PauliWord, constant in value])

        terms = []
        for PauliWord, constant in value:
            new_constant = constant/np.sqrt(factor)
            terms.append((PauliWord, new_constant))

        # anti_commuting_sets[key] = [terms, ('factor', factor)] # can also have *terms

        anti_commuting_sets[key] = {'PauliWords': terms, 'factor': factor}

    return anti_commuting_sets

ll = Get_beta_j_cofactors(anti_commuting_sets)
print(ll[10]['PauliWords'])
print(ll[10]['factor'])


def Get_R_sk_operator(normalised_anti_commuting_sets, S=0): # TODO write function to select 'best' S term!
    X_sk_and_theta_sk={}
    # pick S
    for key in normalised_anti_commuting_sets:
        anti_commuting_set = normalised_anti_commuting_sets[key]['PauliWords']

        if len(anti_commuting_set) > 1:


            k_indexes = [index for index in range(len(anti_commuting_set)) if
                       index != S]

            Op_list = []
            for k in k_indexes:

                X_sk_op =(anti_commuting_set[S], anti_commuting_set[k])

                tan_theta_sk = anti_commuting_set[k][1] / (np.sqrt( anti_commuting_set[S][1] + sum([anti_commuting_set[beta_j][1]**2 for beta_j
                                                                                         in np.arange(1,k, 1)]))) #eqn 16

                theta_sk = np.arctan(tan_theta_sk)

                #Op_list.append((X_sk_op, tan_theta_sk, normalised_anti_commuting_sets[key]['factor']))

                Op_list.append({'X_sk': X_sk_op, 'theta_sk': theta_sk, 'factor': normalised_anti_commuting_sets[key]['factor']})

            X_sk_and_theta_sk.update({key: Op_list})

    return X_sk_and_theta_sk

ww = Get_R_sk_operator(ll, S=0)

print(ww[7][0]['X_sk'])
print(ww[7][0]['theta_sk'])
print(ww[7][0]['factor'])

print(ww[7][1]['X_sk'])
print(ww[7][1]['theta_sk'])
print(ww[7][1]['factor'])


## now have everything for eqn 15 and thus eqn 17!
# 1. apply R_S gate
# 2. Results in P_s being left over :)
# 3. Maybe build this in Cirq!

# do rest in cirq!

def Get_R_S_operator(X_sk_and_theta_sk):
    for key in X_sk_and_theta_sk:
        for i in range(len(X_sk_and_theta_sk[key])):
            X_sk = X_sk_and_theta_sk[key][i]['X_sk']
            theta_sk = X_sk_and_theta_sk[key][i]['theta_sk']
            factor = X_sk_and_theta_sk[key][i]['factor']

            print(X_sk, theta_sk, factor) # TODO build Q circuit from this info!

Get_R_S_operator(ww)


