import pickle
import os
import argparse
import datetime
from quchem_ibm.Qiskit_Chemistry import *
from quchem_ibm.IBM_experiment_functions import *


def Calc_E_M_LCU(output_dict, n_shots):
    E_sim=[output_dict['I_term']]
    T_meas = 0
    exp_record={}
    for index, exp_dict in tqdm(enumerate(output_dict['experiment_dict']), ascii=True, desc='Performing_VQE'):

        final_state=output_dict['state_vector_list'][index]

        if 'Pn' in exp_dict.keys():
            Pn = exp_dict['Pn']
            gamma_l = exp_dict['gamma_l']
            N_ancilla = exp_dict['N_ancilla']

            simulated_dict =final_state.sample_counts(n_shots)

            post_selected_dict_sim = Get_post_selection_counts_DICT_LCU(simulated_dict, N_ancilla)
            exp_val_sim = calc_exp_pauliword(post_selected_dict_sim, Pn)
            E_sim.append(exp_val_sim * gamma_l)
            T_meas+= sum(post_selected_dict_sim.values())

            exp_record[index] = {'measurement_dict': post_selected_dict_sim, #NOTE THIS IS POST_SELECTED!
                                 'Pn': Pn,
                                 'gamma_l': gamma_l}

        else:
            qubitOp = exp_dict['qubitOp']
            coeff = exp_dict['coeff']

            simulated_dict = final_state.sample_counts(n_shots)
            exp_val = calc_exp_pauliword(simulated_dict, qubitOp)
            E_sim.append(exp_val * coeff)
            T_meas+= sum(simulated_dict.values())

            exp_record[index] = {'measurement_dict': simulated_dict,
                                 'qubitOp': qubitOp,
                                 'coeff': coeff}

    return sum(E_sim), T_meas, exp_record

def Calc_E_M_seq_rot(output_dict, n_shots):
    E_sim=[output_dict['I_term']]
    T_meas = 0
    exp_record = {}
    for index, exp_dict in tqdm(enumerate(output_dict['experiment_dict']), ascii=True, desc='Performing_VQE'):

        final_state=output_dict['state_vector_list'][index]

        if 'Ps' in exp_dict.keys():
            Ps = exp_dict['Ps']
            gamma_l = exp_dict['gamma_l']

            simulated_dict =final_state.sample_counts(n_shots)

            exp_val_sim = calc_exp_pauliword(simulated_dict, Ps)
            E_sim.append(exp_val_sim * gamma_l)
            T_meas+= sum(simulated_dict.values())

            exp_record[index] = {'measurement_dict': simulated_dict,
                                 'Ps': Ps,
                                 'gamma_l': gamma_l}
        else:
            qubitOp = exp_dict['qubitOp']
            coeff = exp_dict['coeff']

            simulated_dict = final_state.sample_counts(n_shots)
            exp_val = calc_exp_pauliword(simulated_dict, qubitOp)
            E_sim.append(exp_val * coeff)
            T_meas+= sum(simulated_dict.values())

            exp_record[index] = {'measurement_dict': simulated_dict,
                                 'qubitOp': qubitOp,
                                 'coeff': coeff}

    return sum(E_sim), T_meas, exp_record

def Calc_E_M_STANDARD(output_dict, n_shots):
    E_sim = [output_dict['I_term']]
    T_meas = 0
    exp_record = {}
    for index, exp_dict in tqdm(enumerate(output_dict['experiment_dict']), ascii=True, desc='Performing_VQE'):
        final_state = output_dict['state_vector_list'][index]

        qubitOp = exp_dict['qubitOp']
        coeff = exp_dict['coeff']

        simulated_dict = final_state.sample_counts(n_shots)
        exp_val = calc_exp_pauliword(simulated_dict, qubitOp)
        E_sim.append(exp_val * coeff)
        T_meas += sum(simulated_dict.values())

        exp_record[index] = {'measurement_dict': simulated_dict,
                             'qubitOp': qubitOp,
                             'coeff': coeff}
    return sum(E_sim), T_meas, exp_record


def main(N_repeats):

    file_path_to_LCU_data='/home/ucapar1/Scratch/LiH_wave_exp/molecule=LiH___n_shots=1___method=LCU___time=2020Oct01-164725407693.pickle'
    file_path_to_STANDARD_data='/home/ucapar1/Scratch/LiH_wave_exp/molecule=LiH___n_shots=1___method=standard_VQE___time=2020Oct01-181242726937.pickle'
    file_path_to_seq_rot_data = '/home/ucapar1/Scratch/LiH_wave_exp/molecule=LiH___n_shots=1___method=seq_rot_VQE___time=2020Oct01-164012108374.pickle'

    with open(file_path_to_LCU_data, 'rb') as handle:
        output_data_LCU = pickle.load(handle)

    with open(file_path_to_STANDARD_data, 'rb') as handle:
        output_data_STANDARD = pickle.load(handle)

    with open(file_path_to_seq_rot_data, 'rb') as handle:
        output_data_seq_rot = pickle.load(handle)

    # LCM(630, 102) = 10710
    shot_list= np.arange(107_100, 107_100*1000, 107100*50)

    STANDARD_E_data = np.zeros((len(shot_list), N_repeats))
    STANDARD_M_data = np.zeros((len(shot_list), N_repeats))

    LCU_E_data = np.zeros((len(shot_list), N_repeats))
    LCU_M_data = np.zeros((len(shot_list), N_repeats))

    seq_rot_E_data = np.zeros((len(shot_list), N_repeats))
    seq_rot_M_data = np.zeros((len(shot_list), N_repeats))

    experimental_diff_shot_record={}
    for index, n_shots in enumerate(shot_list):
        exp_repeat_measurement_list_LCU=[]
        exp_repeat_measurement_list_STANDARD=[]
        exp_repeat_measurement_list_SEQ_ROT=[]
        for j in range(N_repeats):
            n_shots_UP = int(n_shots / 102)
            n_shots_STANDARD = int(n_shots / 630)

            E_LCU, M_LCU, exp_record_LCU = Calc_E_M_LCU(output_data_LCU, n_shots_UP)
            LCU_E_data[index, j] = E_LCU
            LCU_M_data[index, j] = M_LCU
            exp_repeat_measurement_list_LCU.append(exp_record_LCU)
            del exp_record_LCU

            E_STANDARD, M_STANDARD, exp_record_STANDARD = Calc_E_M_STANDARD(output_data_STANDARD, n_shots_STANDARD)
            STANDARD_E_data[index, j] = E_STANDARD
            STANDARD_M_data[index, j] = M_STANDARD
            exp_repeat_measurement_list_STANDARD.append(exp_record_STANDARD)
            del exp_record_STANDARD

            E_seq_rot, M_seq_rot, exp_record_SEQ_ROT = Calc_E_M_seq_rot(output_data_seq_rot, n_shots_UP)
            seq_rot_E_data[index, j] = E_seq_rot
            seq_rot_M_data[index, j] = M_seq_rot
            exp_repeat_measurement_list_SEQ_ROT.append(exp_record_SEQ_ROT)
            del exp_record_SEQ_ROT

        experimental_diff_shot_record[n_shots] = {'LCU': exp_repeat_measurement_list_LCU,
                                                  'seq_rot': exp_repeat_measurement_list_SEQ_ROT,
                                                  'standard': exp_repeat_measurement_list_STANDARD}

        output = {
            'shot_list': shot_list,

            'STANDARD_E_data': STANDARD_E_data,
            'STANDARD_M_data': STANDARD_M_data,

            'LCU_E_data': LCU_E_data,
            'LCU_M_data': LCU_M_data,

            'seq_rot_E_data': seq_rot_E_data,
            'seq_rot_M_data': seq_rot_M_data,

            'experiment_data': experimental_diff_shot_record

        }

        time = datetime.datetime.now().strftime('%Y%b%d-%H%M%S%f')
        F_name = 'LiH_simulation_RESULTS_time={}'.format(time)

        #         base_dir = os.path.dirname(os.path.realpath(__file__))
        base_dir = os.getcwd()
        filepath = os.path.join(base_dir, 'LiH_SIMULATION_RESULTS')

        if not os.path.exists(filepath):
            os.makedirs(filepath)

        filepath = os.path.join(filepath, F_name)
        with open(filepath + '.pickle', 'wb') as fhandle:
            pickle.dump(output, fhandle, protocol=pickle.HIGHEST_PROTOCOL)

        print('experiment data saved here: {}'.format(filepath))

        return output

    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument("N_repeats", type=int, help="Number of times experiment repeated (for averaging)")
        args = parser.parse_args()

        main(args.N_repeats)
