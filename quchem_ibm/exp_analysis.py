import pickle
import os
from quchem_ibm.Qiskit_Chemistry import *

def Load_all_pickle_files(data_dir=None, exp_type_str=None):
    if data_dir is None:
        data_dir = os.getcwd() # current_dir
    all_results = []
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.endswith('.pickle'):
                if exp_type_str:
                    if exp_type_str in filename:
                        filepath = os.path.join(data_dir, filename)
                        with open(filepath, 'rb') as handle:
                            all_results.append(pickle.load(handle))
                else:
                    filepath = os.path.join(data_dir, filename)
                    with open(filepath, 'rb') as handle:
                        all_results.append(pickle.load(handle))
    return all_results



def Calc_Energy_seq_rot(experimental_count_list, simulated_count_list, list_experiment_dicts, I_term,
                        standard_meas_filter):
    E_raw_list = [I_term]
    E_sim_list = [I_term]
    E_mit_list = [I_term]

    for index, exp_dict in enumerate(list_experiment_dicts):

        exp_counts_dict = experimental_count_list[index]
        simulated_counts = simulated_count_list[index]
        meas_mit_counts = standard_meas_filter.apply(exp_counts_dict)
        if 'Ps' in exp_dict.keys():
            Ps = exp_dict['Ps']
            gamma_l = exp_dict['gamma_l']

            # raw
            exp_val = calc_exp_pauliword(exp_counts_dict, Ps)
            E_raw_list.append(exp_val * gamma_l)

            # sim
            exp_val_sim = calc_exp_pauliword(simulated_counts, Ps)
            E_sim_list.append(exp_val_sim * gamma_l)

            # mitigated
            exp_val_mit = calc_exp_pauliword(meas_mit_counts, Ps)
            E_mit_list.append(exp_val_mit * gamma_l)
        else:
            qubitOp = exp_dict['qubitOp']
            coeff = exp_dict['coeff']

            # raw
            exp_val = calc_exp_pauliword(exp_counts_dict, qubitOp)
            E_raw_list.append(exp_val * coeff)

            # sim
            exp_val_sim = calc_exp_pauliword(simulated_counts, qubitOp)
            E_sim_list.append(exp_val_sim * coeff)

            # mitigated
            exp_val_mit = calc_exp_pauliword(meas_mit_counts, qubitOp)
            E_mit_list.append(exp_val_mit * coeff)
    return sum(E_raw_list).real, sum(E_sim_list).real, sum(E_mit_list).real


def Calc_Energy_LCU(experimental_meas_list, simulated_meas_list, list_experiment_dicts, I_term,
                    LCU_measurement_filter, standard_meas_filter):
    E_raw_list = [I_term]
    E_sim_list = [I_term]
    E_mit_list = [I_term]

    total_M_raw=0
    total_M_sim=0
    total_M_mit=0

    for index, exp_dict in enumerate(list_experiment_dicts):

        exp_measurements = experimental_meas_list[index]
        simulated_measurements = simulated_meas_list[index]

        if 'Pn' in exp_dict.keys():
            Pn = exp_dict['Pn']
            gamma_l = exp_dict['gamma_l']
            N_ancilla = exp_dict['N_ancilla']

            # raw
            post_selected_dict_raw = Get_post_selection_counts_LCU(exp_measurements, N_ancilla)
            exp_val = calc_exp_pauliword(post_selected_dict_raw, Pn)
            E_raw_list.append(exp_val * gamma_l)
            total_M_raw += sum(post_selected_dict_raw.values())

            # sim
            post_selected_dict_sim = Get_post_selection_counts_LCU(simulated_measurements, N_ancilla)
            exp_val_sim = calc_exp_pauliword(post_selected_dict_sim, Pn)
            E_sim_list.append(exp_val_sim * gamma_l)
            total_M_sim += sum(post_selected_dict_sim.values())

            # mitigated
            raw_counts_dict = Get_post_selection_counts_LCU(exp_measurements, 0)
            mit_counts_dict = LCU_measurement_filter.apply(raw_counts_dict)

            post_selected_dict_mit = Get_post_selection_counts_DICT_LCU(mit_counts_dict, N_ancilla)
            exp_val_mit = calc_exp_pauliword(post_selected_dict_mit, Pn)
            E_mit_list.append(exp_val_mit * gamma_l)
            total_M_mit += sum(post_selected_dict_mit.values())

        else:
            qubitOp = exp_dict['qubitOp']
            coeff = exp_dict['coeff']

            # raw
            post_selected_dict_raw = Get_post_selection_counts_LCU(exp_measurements, 0)
            exp_val = calc_exp_pauliword(post_selected_dict_raw, qubitOp)
            E_raw_list.append(exp_val * coeff)
            total_M_raw += sum(post_selected_dict_raw.values())

            # sim
            post_selected_dict_sim = Get_post_selection_counts_LCU(simulated_measurements, 0)
            exp_val_sim = calc_exp_pauliword(post_selected_dict_sim, qubitOp)
            E_sim_list.append(exp_val_sim * coeff)
            total_M_sim += sum(post_selected_dict_sim.values())

            # mitigated
            post_selected_dict_mit = standard_meas_filter.apply(post_selected_dict_raw)
            exp_val_mit = calc_exp_pauliword(post_selected_dict_mit, qubitOp)
            E_mit_list.append(exp_val_mit * coeff)
            total_M_mit += sum(post_selected_dict_mit.values())

    return sum(E_raw_list).real, sum(E_sim_list).real, sum(E_mit_list).real, total_M_raw, total_M_sim, int(np.floor(total_M_mit))


def Calc_Energy_standard(experimental_count_list, simulated_count_list, list_experiment_dicts, I_term,
                         standard_meas_filter):
    E_raw_list = [I_term]
    E_sim_list = [I_term]
    E_mit_list = [I_term]

    for index, exp_dict in enumerate(list_experiment_dicts):
        exp_counts_dict = experimental_count_list[index]
        simulated_counts = simulated_count_list[index]
        meas_mit_counts = standard_meas_filter.apply(exp_counts_dict)

        qubitOp = exp_dict['qubitOp']
        coeff = exp_dict['coeff']

        # raw
        exp_val = calc_exp_pauliword(exp_counts_dict, qubitOp)
        E_raw_list.append(exp_val * coeff)

        # sim
        exp_val_sim = calc_exp_pauliword(simulated_counts, qubitOp)
        E_sim_list.append(exp_val_sim * coeff)

        # mitigated
        exp_val_mit = calc_exp_pauliword(meas_mit_counts, qubitOp)
        E_mit_list.append(exp_val_mit * coeff)

    return sum(E_raw_list).real, sum(E_sim_list).real, sum(E_mit_list).real