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

    run_experiment_exp_loop_state_vector(
        molecule_name,
        method_name,
        input_data,
        optimization_level=None,
        save=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("method", type=str, help="VQE method (standard_VQE, LCU, seq_rot_VQE")
    # parser.add_argument("IBMQ_backend", type=str, help="name of IBMQ backend device")
    # parser.add_argument("n_shots", type=int, help="number of circuit shots")
    args = parser.parse_args()

    main(args.method)