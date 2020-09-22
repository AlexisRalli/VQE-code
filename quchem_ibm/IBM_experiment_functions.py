import argparse
from qiskit import QuantumRegister, execute, Aer, QuantumCircuit
import datetime
from qiskit.compiler import transpile, assemble
import pickle
import os
from qiskit.providers.ibmq import least_busy
from qiskit import IBMQ
from qiskit.ignis.mitigation.measurement import complete_meas_cal,CompleteMeasFitter
from tqdm import tqdm

def load_IBM_provider(IBM_key=None):
    try:
        IBMQ.load_account()
    except:

        if IBM_key is None:
            IBM_key = '548debe392148d1cb3bdb2019b8f52718d7e3cb473f2d1564a71a6bc6c63fb7251a73ce13875199a5eebacd72369ea4131c527d3993564a32bcc730329286fec'
        IBMQ.save_account(IBM_key, overwrite=True)
        IBMQ.load_account()

    my_provider = IBMQ.get_provider()
    return my_provider

def Get_IBM_backends(IBM_provider, show_least_busy=False):
    print('Available backends: \n{}'.format(list(map(lambda x: x.name(), IBM_provider.backends()))))
    print('FREE backend: qasm_simulator \n')
    if show_least_busy:
        least_busy_device = least_busy(
            IBM_provider.backends(filters=lambda x: x.configuration().n_qubits > 1 and not x.configuration().simulator))
        print('Least busy device: {} \n'.format(least_busy_device.name()))

    print('Press enter to return None for simultion only!')
    IBM_backend = input('Please enter backend name --> ')

    return IBM_backend


def retrieve_job(job_id):
    provider = load_IBM_provider()
    backend = provider.backends.ibmq_qasm_simulator
    retrieved_job = backend.retrieve_job(job_id)
    print(retrieved_job.status())
    return retrieved_job

def Get_Measurement_Filter(n_qubits, backend_name, n_shots, my_provider=None):
    # Error mitigation
    qr_standard = QuantumRegister(n_qubits)
    meas_calibs, state_labels = complete_meas_cal(qr=qr_standard, circlabel='mcal')

    if backend_name == 'qasm_simulator':
        backend = Aer.get_backend(backend_name)
    else:
        backend = my_provider.get_backend(backend_name)

    job_calib = execute(meas_calibs, backend=backend, shots=n_shots)
    cal_results = job_calib.result()
    meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal')

    # Get the filter object
    meas_filter = meas_fitter.filter
    return meas_filter

def run_VQE_unitary_part_experiments(molecule_name, method_name, qc_circuit_dictionary_list, my_provider, n_shots, n_system, I_term,
                                     save=True, optimization_level=3, backend_name=None):

    backend_simulator = Aer.get_backend('qasm_simulator')
    if backend_name:
        backend = my_provider.get_backend(backend_name)

    output = {}

    qc_list = [exp_dict['circuit'] for exp_dict in qc_circuit_dictionary_list if exp_dict['circuit']]
    if backend_name:
        transpiled_circs = transpile(qc_list, backend=backend, optimization_level=optimization_level)

    transpiled_circs_sim=None
    if backend_name:
        if method_name == 'LCU':
            if backend_name:
                qobjs = assemble(transpiled_circs, backend=backend, shots=n_shots, memory=True)
            qobjs_sim= assemble(transpiled_circs, backend=backend_simulator, shots=n_shots, memory=True)
        else:
            if backend_name:
                qobjs = assemble(transpiled_circs, backend=backend, shots=n_shots)
            qobjs_sim = assemble(transpiled_circs, backend=backend_simulator, shots=n_shots)
    else:
        transpiled_circs_sim = transpile(qc_list, backend=backend_simulator, optimization_level=None)
        qobjs_sim = assemble(transpiled_circs_sim, backend=backend_simulator, shots=n_shots,
                             memory=True if method_name == 'LCU' else False)

    transpiled_circs_sim = transpiled_circs if transpiled_circs_sim is None else transpiled_circs_sim
    ## error mitigation
    # https://qiskit.org/textbook/ch-quantum-hardware/measurement-error-mitigation.html
    if backend_name:
        max_shots = 8192
        if method_name == 'LCU':
            LCU_meas_filters=[]
            N_ancilla_per_exp = set([exp_dict['N_ancilla'] for exp_dict in qc_circuit_dictionary_list if 'N_ancilla' in exp_dict.keys()])
            for n_ancilla in N_ancilla_per_exp:
                #fitter is n_ancilla + n_system!
                LCU_meas_filters.append({'N_ancilla': n_ancilla, 'filter': Get_Measurement_Filter(n_ancilla+n_system, backend_name, max_shots, my_provider=my_provider)})

        stnd_meas_filter = Get_Measurement_Filter(n_system, backend_name, max_shots, my_provider=my_provider)
    #### end error mitigation

    output['start_time'] = datetime.datetime.now()
    print('## running job ## \n')
    if backend_name:
        job = backend.run(qobjs)
    sim_job = backend_simulator.run(qobjs_sim)

    # if (job.status().name != 'DONE') and (backend.configuration().simulator is not True):
    #     try:
    #         print('queue position = ', job.queue_position())
    #     except:
    #         print('could not find queue position')
    ###
    # while job.status().name != 'DONE':
    #     try:
    #         print('queue position = ', job.queue_position())
    #         time.sleep(60*5)
    #     except:
    #         break

    print('## job finished ## \n')

    output['end_time'] = datetime.datetime.now()
    output['method'] = method_name
    output['n_shots'] = n_shots
    output['I_term'] = I_term
    output['q_circuit_list'] = qc_list # from theory
    output['SIM_transpiled_q_circuit_list'] = transpiled_circs_sim #transpiled_circs_sim
    output['experiment_dict'] = qc_circuit_dictionary_list
    output['SIM_count_list_raw'] = [sim_job.result().get_counts(i) for i in range(len(transpiled_circs_sim))]
    if backend_name:
        output['jobID'] = job.job_id()
        output['EXP_transpiled_q_circuit_list'] = transpiled_circs  # for chip
        output['EXP_backend'] = backend.name()
        # output['EXP_count_list_raw'] = [job.result().get_counts(transpiled_circ) for transpiled_circ in
        #                            transpiled_circs]
        output['EXP_count_list_raw'] = [job.result().get_counts(i) for i in range(len(transpiled_circs))]
        output['standard_measurement_filter'] = stnd_meas_filter

    if method_name == 'LCU':
        # output['SIM_memory_raw'] = [sim_job.result().get_memory(transpiled_circ) for transpiled_circ in
        #                             transpiled_circs_sim]
        output['SIM_memory_raw'] = [sim_job.result().get_memory(i) for i in range(len(transpiled_circs_sim))]
        if backend_name:
            # output['EXP_memory_raw'] = [job.result().get_memory(transpiled_circs[transpiled_circ]) for transpiled_circ in
            #                         transpiled_circs]
            output['EXP_memory_raw'] = [job.result().get_memory(i) for i in range(len(transpiled_circs))]
            output['LCU_measurement_filters'] = LCU_meas_filters

    if save:
        print('## saving job ## \n')
        time = datetime.datetime.now().strftime('%Y%b%d-%H%M%S%f')
        F_name = 'molecule={}___n_shots={}___method={}___time={}'.format(molecule_name, n_shots, method_name, time)

        #         base_dir = os.path.dirname(os.path.realpath(__file__))
        base_dir = os.getcwd()
        data_dir = os.path.join(base_dir, 'Data')

        if backend_name:
            filepath = os.path.join(data_dir, 'device')
        else:
            filepath = os.path.join(data_dir, 'simulator')

        if not os.path.exists(filepath):
            os.makedirs(filepath)

        filepath = os.path.join(filepath, F_name)
        with open(filepath + '.pickle', 'wb') as fhandle:
            pickle.dump(output, fhandle, protocol=pickle.HIGHEST_PROTOCOL)

        print('experiment data saved here: {}'.format(filepath))

    return output

def run_experiment_exp_loop(molecule_name, method_name, my_provider, backend_name, INPUT_dict, shot_list, n_system,
                            optimization_level=3):

    if method_name == 'standard_VQE':
        qc_circuit_dictionary_list = INPUT_dict['standard_VQE_circuits']
        I_term = INPUT_dict['standard_I_term']

    elif method_name== 'seq_rot_VQE':
        qc_circuit_dictionary_list = INPUT_dict['Seq_Rot_VQE_circuits']
        I_term = INPUT_dict['Seq_Rot_I_term']

    elif method_name== 'LCU':
        qc_circuit_dictionary_list = INPUT_dict['LCU_VQE_circuits']
        I_term = INPUT_dict['LCU_I_term']
    else:
        raise ValueError('unknown method {}'.format(method_name))

    for n_shots in tqdm(shot_list, ascii=True, desc='Running experiments'):
    # for n_shots in shot_list:
        run_VQE_unitary_part_experiments(molecule_name,
                                        method_name,
                                        qc_circuit_dictionary_list,
                                        my_provider,
                                        n_shots,
                                        n_system,
                                        I_term,
                                        save = True,
                                        optimization_level = optimization_level,
                                        backend_name = backend_name)



def run_VQE_unitary_part_experiments_QASM(molecule_name, method_name, qc_circuit_dictionary_list, my_provider, n_shots, I_term, backend_name,
                                     save=True, optimization_level=3):

    backend = my_provider.get_backend(backend_name)

    output = {}

    # QASM PART!
    qc_list = [QuantumCircuit.from_qasm_str(exp_dict['circuit']) for exp_dict in tqdm(qc_circuit_dictionary_list, ascii=True, desc='Getting Q Circuits') if exp_dict['circuit']]
    # qc_list = [QuantumCircuit.from_qasm_str(exp_dict['circuit']) for exp_dict in qc_circuit_dictionary_list if exp_dict['circuit']]
    transpiled_circs = transpile(qc_list, backend=backend, optimization_level=optimization_level)
    del qc_list

    if method_name == 'LCU':
        qobjs = assemble(transpiled_circs, backend=backend, shots=n_shots, memory=True)
    else:
        qobjs = assemble(transpiled_circs, backend=backend, shots=n_shots)

    transpiled_circs_length= len(transpiled_circs)
    del transpiled_circs

    # ## error mitigation
    # # https://qiskit.org/textbook/ch-quantum-hardware/measurement-error-mitigation.html
    # if backend_name:
    #     max_shots = 8192
    #     if method_name == 'LCU':
    #         LCU_meas_filters=[]
    #         N_ancilla_per_exp = set([exp_dict['N_ancilla'] for exp_dict in qc_circuit_dictionary_list if 'N_ancilla' in exp_dict.keys()])
    #         for n_ancilla in N_ancilla_per_exp:
    #             #fitter is n_ancilla + n_system!
    #             LCU_meas_filters.append({'N_ancilla': n_ancilla, 'filter': Get_Measurement_Filter(n_ancilla+n_system, backend_name, max_shots, my_provider=my_provider)})
    #
    #     stnd_meas_filter = Get_Measurement_Filter(n_system, backend_name, max_shots, my_provider=my_provider)
    # #### end error mitigation

    output['start_time'] = datetime.datetime.now()
    print('## running job ## \n')
    job = backend.run(qobjs)

    # if (job.status().name != 'DONE') and (backend.configuration().simulator is not True):
    #     try:
    #         print('queue position = ', job.queue_position())
    #     except:
    #         print('could not find queue position')
    ###
    # while job.status().name != 'DONE':
    #     try:
    #         print('queue position = ', job.queue_position())
    #         time.sleep(60*5)
    #     except:
    #         break

    print('## job finished ## \n')

    output['end_time'] = datetime.datetime.now()
    output['method'] = method_name
    output['n_shots'] = n_shots
    output['I_term'] = I_term
    # output['q_circuit_list'] = qc_list # from theory
    # output['SIM_transpiled_q_circuit_list'] = transpiled_circs_sim #transpiled_circs_sim
    output['experiment_dict'] = qc_circuit_dictionary_list
    # output['SIM_count_list_raw'] = [sim_job.result().get_counts(i) for i in range(len(transpiled_circs_sim))]
    if backend_name:
        output['jobID'] = job.job_id()
        # output['EXP_transpiled_q_circuit_list'] = transpiled_circs  # for chip
        output['EXP_backend'] = backend.name()
        # output['EXP_count_list_raw'] = [job.result().get_counts(transpiled_circ) for transpiled_circ in
        #                            transpiled_circs]
        output['EXP_count_list_raw'] = [job.result().get_counts(i) for i in range(transpiled_circs_length)]
        # output['standard_measurement_filter'] = stnd_meas_filter

    if method_name == 'LCU':
        # output['SIM_memory_raw'] = [sim_job.result().get_memory(transpiled_circ) for transpiled_circ in
        #                             transpiled_circs_sim]
        # output['SIM_memory_raw'] = [sim_job.result().get_memory(i) for i in range(len(transpiled_circs_sim))]
        if backend_name:
            # output['EXP_memory_raw'] = [job.result().get_memory(transpiled_circs[transpiled_circ]) for transpiled_circ in
            #                         transpiled_circs]
            output['EXP_memory_raw'] = [job.result().get_memory(i) for i in range(transpiled_circs_length)]
            # output['LCU_measurement_filters'] = LCU_meas_filters

    if save:
        print('## saving job ## \n')
        time = datetime.datetime.now().strftime('%Y%b%d-%H%M%S%f')
        F_name = 'molecule={}___n_shots={}___method={}___time={}'.format(molecule_name, n_shots, method_name, time)

        #         base_dir = os.path.dirname(os.path.realpath(__file__))
        base_dir = os.getcwd()
        filepath = os.path.join(base_dir, 'IBM_qasm_simulator')

        if not os.path.exists(filepath):
            os.makedirs(filepath)

        filepath = os.path.join(filepath, F_name)
        with open(filepath + '.pickle', 'wb') as fhandle:
            pickle.dump(output, fhandle, protocol=pickle.HIGHEST_PROTOCOL)

        print('experiment data saved here: {}'.format(filepath))

    return output




def run_experiment_exp_loop_QASM(molecule_name, method_name, my_provider, backend_name, INPUT_dict, shot_list,
                            optimization_level=3):

    if method_name== 'standard_VQE':
        qc_circuit_dictionary_list = INPUT_dict['standard_VQE_circuits']
        I_term = INPUT_dict['standard_I_term']

    elif method_name== 'seq_rot_VQE':
        qc_circuit_dictionary_list = INPUT_dict['Seq_Rot_VQE_circuits']
        I_term = INPUT_dict['Seq_Rot_I_term']

    elif method_name== 'LCU':
        qc_circuit_dictionary_list = INPUT_dict['LCU_VQE_circuits']
        I_term = INPUT_dict['LCU_I_term']
    else:
        raise ValueError('unknown method {}'.format(method_name))


    for n_shots in tqdm(shot_list, ascii=True, desc='Running experiments'):
    # for n_shots in shot_list:
        run_VQE_unitary_part_experiments_QASM(molecule_name,
                                              method_name,
                                              qc_circuit_dictionary_list,
                                              my_provider,
                                              n_shots,
                                              I_term,
                                              backend_name,
                                              save=True,
                                              optimization_level=optimization_level)

