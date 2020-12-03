import pandas as pd
from quchem.LCU_method import *

def latex_table_Hamiltonian(Hamiltonian_sets, file_name_str):
    H_sl_list = []
    for key in Hamiltonian_sets:
        H_sl = []
        for qubitOp in Hamiltonian_sets[key]:
            #             print(qubitOp.__str__())
            H_sl.append(qubitOp.__str__())

        H_sl_list.append(H_sl)

    df = pd.DataFrame({'l index': list(Hamiltonian_sets.keys()),
                       '$H_{S_{l}}$': H_sl_list})
    pd.set_option('display.max_colwidth', 1000)

    file_name = '{}.tex'.format(file_name_str)
    with open(file_name, 'w') as f_handle:
        f_handle.write(df.to_latex(index=False))

    print(df.to_latex(index=False, multirow=True))

# import pandas as pd

# def latex_table_Hamiltonian_normalised(Hamiltonian_sets, file_name_str):
#     H_sl_list=[]

#     H_sl_list_norm=[]
#     gamma_l_list=[]
#     for key in Hamiltonian_sets:
#         H_sl=[]
#         H_sl_norm=[]
#         normalised_term_dict = Get_beta_j_cofactors(Hamiltonian_sets[key])
#         for index, qubitOp in enumerate(normalised_term_dict['PauliWords']):
# #             print(qubitOp.__str__())
#             H_sl_norm.append(qubitOp.__str__())
#             H_sl.append(Hamiltonian_sets[key][index].__str__())

#         H_sl_list.append(H_sl)

#         H_sl_list_norm.append(H_sl_norm)
#         gamma_l_list.append(normalised_term_dict['gamma_l'])

#     df = pd.DataFrame({'l index': list(Hamiltonian_sets.keys()),
#                    '$H_{S_{l}}$': H_sl_list,
#                    '$\gamma_{l}$': gamma_l_list,
#                       '$\frac{H_{S_{l}}}{\gamma_{l}}$': H_sl_list_norm,
#                       })

#     pd.set_option('display.max_colwidth', 1000)

#     file_name = '{}.tex'.format(file_name_str)
#     with open(file_name, 'w') as f_handle:
#         f_handle.write(df.to_latex(index=False))

#     print(df.to_latex(index=False, multirow = True))

# latex_table_Hamiltonian_normalised(anti_commuting_sets, 'Latex_table')

def latex_table_seq_rot(Hamiltonian_sets, file_name_str, S_index):
    gamma_l_list = []
    X_sk_list = []
    theta_sk_list = []
    key_list = []
    H_sl_list_norm = []
    Pn_list=[]

    for key in Hamiltonian_sets:
        if len(Hamiltonian_sets[key]) > 1:
            key_list.append(key)
            X_sk_ops = []
            theta_sk_ops = []
            X_sk_theta_sk, normalised_FULL_set, Ps, gamma_l = Get_Xsk_op_list(Hamiltonian_sets[key], S_index)
            for X_sk_theta_sk_tuple in X_sk_theta_sk:
                qubitOp, theta_sk = X_sk_theta_sk_tuple
                X_sk_ops.append(qubitOp.__str__())
                theta_sk_ops.append(theta_sk.real)
            Pn_list.append(Ps)

            X_sk_list.append(X_sk_ops)
            theta_sk_list.append(theta_sk_ops)

            H_sl_list_norm.append([term.__str__() for term in normalised_FULL_set['PauliWords']])
            gamma_l_list.append(normalised_FULL_set['gamma_l'].real)

    df = pd.DataFrame({'l index': key_list,
                       '$\gamma_{l}$': gamma_l_list,
                       '$\frac{H_{S_{l}}}{\gamma_{l}}$': H_sl_list_norm,
                       'Pn': Pn_list,
                       '$\mathcal{X_{nk}}$': X_sk_list,
                       '$\theta_{nk}}$': theta_sk_list
                       })

    pd.set_option('display.max_colwidth', 1000)

    file_name = '{}.tex'.format(file_name_str)
    with open(file_name, 'w') as f_handle:
        f_handle.write(df.to_latex(index=False))

    print(df.to_latex(index=False, multirow=True))

def Get_R_op_list_for_table(anti_commuting_set, N_index, check_operator=False):
    """
    copied and tweaked
     """

    # 𝛾_𝑙 ∑ 𝛽_𝑗 𝑃_𝑗
    normalised_FULL_set = Get_beta_j_cofactors(anti_commuting_set)
    gamma_l = normalised_FULL_set['gamma_l']

    norm_FULL_set = normalised_FULL_set['PauliWords'].copy()

    # 𝛽_n 𝑃_n
    qubitOp_Pn_beta_n = norm_FULL_set.pop(N_index)

    # Ω_𝑙 ∑ 𝛿_k 𝑃_k  ... note this doesn't contain 𝛽_n 𝑃_n
    H_n_1 = Get_beta_j_cofactors(norm_FULL_set)
    Omega_l = H_n_1['gamma_l'].real

    ##

    # cos(𝜙_{𝑛−1}) =𝛽_𝑛
    phi_n_1 = np.arccos(list(qubitOp_Pn_beta_n.terms.values())[0])

    # require sin(𝜙_{𝑛−1}) to be positive...
    # this uses CAST diagram to ensure the sign term is positive and cos term has correct sign (can be negative)
    if (phi_n_1 > np.pi):
        # ^ as sin phi_n_1 must be positive phi_n_1 CANNOT be larger than 180 degrees!
        phi_n_1 = 2 * np.pi - phi_n_1
        print('correct quadrant found!!!')

    #     𝑅=exp(−𝑖𝛼 X/2)=cos(𝛼/2)𝟙 − 𝑖sin(𝛼/2)X = cos(𝛼/2)𝟙 − 𝑖sin(𝛼/2)(i∑𝛿𝑘 𝑃𝑘𝑃𝑛)
    #     𝑅=exp(−𝑖𝛼 X/2)=cos(𝛼/2)𝟙 − 𝑖sin(𝛼/2)X = cos(𝛼/2)𝟙 + sin(𝛼/2)(∑𝛿𝑘 𝑃𝑘𝑃𝑛) #<--- note sign here!
    Pn = QubitOperator(list(qubitOp_Pn_beta_n.terms.keys())[0],
                       1)  # np.sign(list(qubitOp_Pn_beta_n.terms.values())[0]))

    alpha = phi_n_1.copy()
    #     print('alpha/2 =', (alpha/(2*np.pi))*360/2)

    I_term = QubitOperator('', np.cos(alpha / 2))
    R_linear_comb_list = [I_term]

    sin_term = -np.sin(alpha / 2)

    for qubitOp_Pk in H_n_1['PauliWords']:
        PkPn = qubitOp_Pk * Pn
        R_linear_comb_list.append(sin_term * PkPn)

    Bk_Pkn=[]
    for qubitOp_Pk in H_n_1['PauliWords']:
        PkPn = qubitOp_Pk * Pn
        Bk_Pkn.append(PkPn)

    if not np.isclose(sum(np.absolute(list(qubitOp.terms.values())[0]) ** 2 for qubitOp in R_linear_comb_list), 1):
        raise ValueError(
            'normalisation of X operator incorrect: {}'.format(sum(np.absolute(list(qubitOp.terms.values())[0]) ** 2
                                                                   for qubitOp in R_linear_comb_list)))
    if check_operator:
    # #     # 𝐻𝑛= B𝑛𝑃𝑛+ Ω 𝑙∑𝛿𝑃𝑗
    # #     print('Hn =',qubitOp_Pn_beta_n, '+', Omega_l,' * ', H_n_1['PauliWords'])
    # #     #𝐻𝑛= cos(𝜙_{n-1}) Pn + sin(𝜙_{n-1}) H_{n_1 }
    # #     print('Hn =',np.cos(phi_n_1),Pn, '+', np.sin(phi_n_1),' * ', H_n_1['PauliWords'])
        Hn_list = [qubitOp_Pn_beta_n] + [Omega_l* op for op in  H_n_1['PauliWords']]
    #
    # #     print('')
    # #     print('R = ', R_linear_comb_list)
    # #     #R= cos(𝛼/2)𝟙-sin(𝛼/2)(∑𝛿_{𝑘}𝑃_{𝑘𝑛})
    # #     print('R = ', np.cos(alpha/2), 'I', '+',np.sin(alpha/2), [dkPk*Pn for dkPk in H_n_1['PauliWords']])

        ### CHECKING need to comment out as expensive!
        R = QubitOperator()
        for op in R_linear_comb_list:
            R += op

        R_dag = QubitOperator()
        for op in R:
            if list(op.terms.keys())[0]==():
                R_dag+= QubitOperator('', list(op.terms.values())[0])
            else:
                R_dag+=op*-1   #  note sign!!!

        H_n = QubitOperator()
        for op in Hn_list:
            H_n += op

        print('Pn= R*H_n*R_dag ')
        print('Pn=', Pn)
        print('R*H_n*R_dag = ', R * H_n * R_dag)
            # print('Pn= R*H_n*R_dag ', Pn, ' = ', R*H_n*R_dag)
        #     print('H_n= R_dag*Pn*R ', H_n, ' = ', R_dag*Pn*R)

    return normalised_FULL_set['PauliWords'], gamma_l, qubitOp_Pn_beta_n, Omega_l, H_n_1['PauliWords'], R_linear_comb_list, alpha, Bk_Pkn

def latex_table_LCU(Hamiltonian_sets, file_name_str, N_index):
    gamma_l_list = []
    omega_l_list = []
    beta_n_pauli_n_list = []
    R_list = []
    key_list = []
    H_n_1_list = []
    alpha_list=[]
    Bk_Pkn_list=[]

    H_sl_list_norm = []

    for key in Hamiltonian_sets:
        if len(Hamiltonian_sets[key]) > 1:
            normalised_FULL_set, gamma_l, qubitOp_Pn_beta_n, Omega_l, H_n_1, R_linear_comb_list, alpha, Bk_Pkn = Get_R_op_list_for_table(
                Hamiltonian_sets[key], N_index, check_operator=False)

            key_list.append(key)
            H_sl_list_norm.append([term.__str__() for term in normalised_FULL_set])
            gamma_l_list.append(gamma_l.real)
            omega_l_list.append(Omega_l.real)
            H_n_1_list.append([qubitOp.__str__() for qubitOp in H_n_1])
            beta_n_pauli_n_list.append(qubitOp_Pn_beta_n)
            R_list.append([qubitOp.__str__() for qubitOp in R_linear_comb_list])
            alpha_list.append(alpha.real)
            Bk_Pkn_list.append(Bk_Pkn)

    df = pd.DataFrame({'l index': key_list,
                       '$\gamma_{l}$': gamma_l_list,
                       '$\frac{H_{S_{l}}}{\gamma_{l}}$': H_sl_list_norm,
                       '$\beta_{n}P_{n}}$': beta_n_pauli_n_list,
                       '$\Omega_{l}$': omega_l_list,
                       '$\sum \delta_{j}^{(l)}P_{j}^{(l)}$': H_n_1_list,
                       'alpha': alpha_list,
                       # 'Bk_Pkn': Bk_Pkn_list,
                       })

    pd.set_option('display.max_colwidth', 1000)

    file_name = '{}.tex'.format(file_name_str)
    with open(file_name, 'w') as f_handle:
        f_handle.write(df.to_latex(index=False))

    print(df.to_latex(index=False, multirow=True))

def latex_table_LCU_R_op(Hamiltonian_sets, file_name_str, N_index):
    R_list = []
    key_list = []

    R_list_real = []
    phase_list = []
    ancilla_amplitude_list = []
    l1_norm_list = []

    for key in Hamiltonian_sets:
        if len(Hamiltonian_sets[key]) > 1:
            normalised_FULL_set, gamma_l, qubitOp_Pn_beta_n, Omega_l, H_n_1, R_linear_comb_list, alpha, Bk_Pkn = Get_R_op_list_for_table(
                Hamiltonian_sets[key], N_index, check_operator=False)

            R_linear_comb_corrected_phase, R_linear_comb_correction_values, ancilla_amplitudes, l1_norm = absorb_complex_phases(
                R_linear_comb_list)

            key_list.append(key)
            R_list.append([qubitOp.__str__() for qubitOp in R_linear_comb_list])
            R_list_real.append([qubitOp.__str__() for qubitOp in R_linear_comb_corrected_phase])
            phase_list.append([phase for phase in R_linear_comb_correction_values])
            ancilla_amplitude_list.append(ancilla_amplitudes)
            l1_norm_list.append(l1_norm)

    df = pd.DataFrame({'l index': key_list,
                       '$R$': R_list,
                       'l1 norm': l1_norm_list,
                       'R real': R_list_real,
                       'phase correction': phase_list,
                       'ancilla amplitudes': ancilla_amplitude_list,
                       })

    pd.set_option('display.max_colwidth', 1000)

    file_name = '{}.tex'.format(file_name_str)
    with open(file_name, 'w') as f_handle:
        f_handle.write(df.to_latex(index=False))

    print(df.to_latex(index=False, multirow=True))

