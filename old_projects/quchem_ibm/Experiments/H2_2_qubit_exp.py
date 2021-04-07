from quchem_ibm.IBM_experiment_functions import *
import pickle
import os
import argparse
import numpy as np


def main(method_name):
    molecule_name='H2'

    ## Load input data
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, 'Input_data')
    input_file = os.path.join(data_dir, 'H2_bravyi_kitaev_2_qubit_experiment_time=2020Sep21-162239536536.pickle')
    with open(input_file, 'rb') as handle:
        input_data = pickle.load(handle)

    ## Get IBM account
    my_provider = load_IBM_provider()
    # IBM_backend = Get_IBM_backends(my_provider, show_least_busy=False)
    IBM_backend = 'ibmqx2'
    # IBM_backend = None


    # # goes up to 4914 for standard VQE and 8190 for unitary partitioning!
    # shot_experiment_list = np.arange(1190 * 3, (8190 * 3) + 1, 2625, dtype=int)
    # if method_name == 'standard_VQE':
    #     shot_list=shot_experiment_list/len(input_data['standard_VQE_circuits'])
    # else:
    #     shot_list = shot_experiment_list/len(input_data['Seq_Rot_VQE_circuits'])

    ## array([1190., 2065., 2940., 3815., 4690., 5565., 6440., 7315., 8190.])
    ## array([ 714., 1239., 1764., 2289., 2814., 3339., 3864., 4389., 4914.])

    if method_name == 'standard_VQE':
        shot_list = [714 for _ in range(10)]
        # shot_list = [*[2814 for _ in range(10)], *[2289 for _ in range(10)],*[1764 for _ in range(10)]]#,
                     # *[1239 for _ in range(10)], *[714 for _ in range(10)]]
        # shot_list=[4914 for _ in range(10)] # max for standard vqe
    else:
        shot_list=[1190 for _ in range(5)]
        # shot_list = [*[4690 for _ in range(10)], *[3815 for _ in range(10)],*[2940 for _ in range(10)]]#,
                     # *[2065 for _ in range(10)], *[1190 for _ in range(10)]]
        # shot_list=[8190 for _ in range(10)] # max for unitary part

    n_system_qubits = input_data['n_system_qubits']

    run_experiment_exp_loop(molecule_name,
                            method_name,
                                 my_provider,
                                 IBM_backend,
                                 input_data,
                                 shot_list,
                                 n_system_qubits,
                                 optimization_level=3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("method", type=str, help="VQE method (standard_VQE, LCU, seq_rot_VQE")
    # parser.add_argument("IBMQ_backend", type=str, help="name of IBMQ backend device")
    # parser.add_argument("n_shots", type=int, help="number of circuit shots")
    args = parser.parse_args()

    main(args.method)


