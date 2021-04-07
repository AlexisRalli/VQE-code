from quchem_ibm.IBM_experiment_functions import *
import pickle
import os
import argparse


def main(method_name):
    molecule_name='LiH'
    ## Load input data
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, 'Input_data')
    input_file = os.path.join(data_dir, 'LiH_bravyi_kitaev_12_qubit_experiment_time=2020Sep21-132537266776.pickle')
    with open(input_file, 'rb') as handle:
        input_data = pickle.load(handle)

    ## Get IBM account
    my_provider = load_IBM_provider()
    # IBM_backend = Get_IBM_backends(my_provider, show_least_busy=False)
    IBM_backend = 'ibmq_qasm_simulator'

    # LCM(102,630)=10710

    # (10710*475)/630 = 1326
    # (10710*78)/102 = 8190
    # shot_list=[8190 for _ in range (10)]
    # shot_list=[1326 for _ in range (10)]
    # shot_list = [8190]
    shot_list = [1326]

    run_experiment_exp_loop_CLOUD(molecule_name,
                                 method_name,
                                 my_provider,
                                 IBM_backend,
                                 input_data,
                                 shot_list,
                                 optimization_level=3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("method", type=str, help="VQE method (standard_VQE, LCU, seq_rot_VQE")
    # parser.add_argument("IBMQ_backend", type=str, help="name of IBMQ backend device")
    # parser.add_argument("n_shots", type=int, help="number of circuit shots")
    args = parser.parse_args()

    main(args.method)