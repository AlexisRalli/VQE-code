import pickle
import os
import numpy as np
import datetime
import argparse
from quchem_ibm.exp_analysis import *
from tqdm import tqdm

def dict_of_M_to_list(M_dict, PauliOP):
    P_Qubit_list, _ = zip(*(list(*PauliOP.terms.keys())))

    list_of_M_bitstrings = None
    for bit_string, N_obtained in M_dict.items():

        M_string = np.take(list(bit_string[::-1]), P_Qubit_list)  # only take terms measured! Note bitstring reversed!

        array_meas = np.repeat(''.join(M_string), N_obtained)
        if list_of_M_bitstrings is None:
            list_of_M_bitstrings = array_meas
        else:
            list_of_M_bitstrings = np.hstack((list_of_M_bitstrings, array_meas))

    # randomly shuffle (seed means outcome will always be the SAME!)
    np.random.seed(42)
    np.random.shuffle(list_of_M_bitstrings)

    return list_of_M_bitstrings

def Get_Hist_data(Histogram_data, I_term):
    E_list=[]
    for m_index in tqdm(range(Histogram_data[0]['Measurements'].shape[0])):
        E=I_term
        for M_dict_key in Histogram_data:
            coeff = Histogram_data[M_dict_key]['coeff']
            parity =  1 if sum(map(int, Histogram_data[M_dict_key]['Measurements'][m_index])) % 2 == 0 else -1
            E+=coeff*parity
        E_list.append(E)
    return E_list

def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp( - ((x - mean) / standard_deviation) ** 2)

def main():
    # # # input for exp
    # base_dir = os.getcwd()
    # input_file = os.path.join(base_dir, 'LiH_simulation_RESULTS_time=2020Oct07-163210198971.pickle')
    file_path_data = '/home/ucapar1/Scratch/LiH_Analysis/molecule=LiH___n_shots=1___method=LCU___time=2020Oct01-164725407693.pickle'
    with open(file_path_data, 'rb') as handle:
        LiH_data = pickle.load(handle)

    LCU_Hist_data_sim = {}
    for shot_key in list(LiH_data['experiment_data'].keys()):

        for exp_instance in LiH_data['experiment_data'][shot_key]['LCU']:  # each exp repeated 10 times!
            for exp_dict_key in exp_instance:

                exp_dict = exp_instance[exp_dict_key]

                if 'Pn' in exp_dict.keys():
                    P = exp_dict['Pn']
                    coeff = exp_dict['gamma_l']

                    measured_dict_sim = exp_dict['measurement_dict']  # post selection already done!

                else:
                    P = exp_dict['qubitOp']
                    coeff = exp_dict['coeff']
                    measured_dict_sim = exp_dict['measurement_dict']

                M_list_sim = dict_of_M_to_list(measured_dict_sim, P)

                if exp_dict_key in LCU_Hist_data_sim.keys():
                    LCU_Hist_data_sim[exp_dict_key] = {'P': P, 'coeff': coeff, 'Measurements': np.hstack(
                        (LCU_Hist_data_sim[exp_dict_key]['Measurements'], M_list_sim))}
                else:
                    LCU_Hist_data_sim[exp_dict_key] = {'P': P, 'coeff': coeff, 'Measurements': M_list_sim}

    # as probablistic need to all be same shape (cannot have more measurements of one term)
    fewest_succ_shots_sim = min([LCU_Hist_data_sim[key]['Measurements'].shape[0] for key in LCU_Hist_data_sim])
    for key in LCU_Hist_data_sim.keys():
        LCU_Hist_data_sim[key]['Measurements'] = LCU_Hist_data_sim[key]['Measurements'][:fewest_succ_shots_sim]

    print('LCU_Hist_data finished')

    SEQ_ROT_Hist_data_sim = {}
    for shot_key in list(LiH_data['experiment_data'].keys()):

        for exp_instance in LiH_data['experiment_data'][shot_key]['seq_rot']:  # each exp repeated 10 times!
            for exp_dict_key in exp_instance:

                exp_dict = exp_instance[exp_dict_key]

                if 'Ps' in exp_dict.keys():
                    P = exp_dict['Ps']
                    coeff = exp_dict['gamma_l']

                    measured_dict_sim = exp_dict['measurement_dict']

                else:
                    P = exp_dict['qubitOp']
                    coeff = exp_dict['coeff']
                    measured_dict_sim = exp_dict['measurement_dict']

                M_list_sim = dict_of_M_to_list(measured_dict_sim, P)

                if exp_dict_key in SEQ_ROT_Hist_data_sim.keys():
                    SEQ_ROT_Hist_data_sim[exp_dict_key] = {'P': P, 'coeff': coeff, 'Measurements': np.hstack(
                        (SEQ_ROT_Hist_data_sim[exp_dict_key]['Measurements'], M_list_sim))}
                else:
                    SEQ_ROT_Hist_data_sim[exp_dict_key] = {'P': P, 'coeff': coeff, 'Measurements': M_list_sim}

    print('SEQ_ROT_Hist_data finished')

    STANDARD_Hist_data_sim = {}
    for shot_key in list(LiH_data['experiment_data'].keys()):

        for exp_instance in LiH_data['experiment_data'][shot_key]['standard']:  # each exp repeated 10 times!
            for exp_dict_key in exp_instance:

                P = exp_dict['qubitOp']
                coeff = exp_dict['coeff']
                measured_dict_sim = exp_dict['measurement_dict']

                M_list_sim = dict_of_M_to_list(measured_dict_sim, P)

                if exp_dict_key in STANDARD_Hist_data_sim.keys():
                    STANDARD_Hist_data_sim[exp_dict_key] = {'P': P, 'coeff': coeff, 'Measurements': np.hstack(
                        (STANDARD_Hist_data_sim[exp_dict_key]['Measurements'], M_list_sim))}
                else:
                    STANDARD_Hist_data_sim[exp_dict_key] = {'P': P, 'coeff': coeff, 'Measurements': M_list_sim}

    print('STANDARD_Hist_data finished')

    I_term = -4.142299396835105
    E_LCU_list_sim=Get_Hist_data(LCU_Hist_data_sim, I_term)
    E_list_SEQ_ROT_sim=Get_Hist_data(SEQ_ROT_Hist_data_sim, I_term)
    E_list_STANDARD_sim=Get_Hist_data(STANDARD_Hist_data_sim, I_term)

    output_sinlge_shots={
        'Single_shot_LCU': E_LCU_list_sim,
         'Single_shot_SEQ_ROT': E_list_SEQ_ROT_sim,
        'Single_shot_STANDARD': E_list_STANDARD_sim
    }

    output_hist_data={
        'LCU_hist_data': LCU_Hist_data_sim,
        'SEQ_ROT_hist_data': SEQ_ROT_Hist_data_sim,
        'STANDARD_hist_data': STANDARD_Hist_data_sim
    }


    time = datetime.datetime.now().strftime('%Y%b%d-%H%M%S%f')
    F_name = 'LiH_Analysis_time={}'.format(time)
    base_dir = os.getcwd()
    filepath = os.path.join(base_dir, F_name)
    with open(filepath +'single_shots' + '.pickle', 'wb') as fhandle:
        pickle.dump(output_sinlge_shots, fhandle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(filepath +'Hist_data' + '.pickle', 'wb') as fhandle:
        pickle.dump(output_hist_data, fhandle, protocol=pickle.HIGHEST_PROTOCOL)

    print('experiment data saved here: {}'.format(filepath))

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("data_file_loc", type=str, help="Location of experiment data")
    # # parser.add_argument("IBMQ_backend", type=str, help="name of IBMQ backend device")
    # # parser.add_argument("n_shots", type=int, help="number of circuit shots")
    # args = parser.parse_args()
    # main(args.data_file_loc)
    main()
