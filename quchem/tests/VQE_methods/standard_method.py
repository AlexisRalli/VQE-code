if __name__ == '__main__':
    from VQE_methods.Simulating_Quantum_Circuit import *
    from VQE_methods.quantum_circuit_functions import *
    from VQE_methods.Ansatz_Generator_Functions import *
else:
    from tests.VQE_methods.Simulating_Quantum_Circuit import *
    from tests.VQE_methods.quantum_circuit_functions import *
    from tests.VQE_methods.Ansatz_Generator_Functions import *

def Get_PauliWord_strings_and_constant(PauliWords, constants):
    """

    :param PauliWords: list of lists of PauliWords
    :type PauliWords: list
    e.g.
    [
        [(0, 'I'), (1, 'I'), (2, 'I'), (3, 'I')],
        [(0, 'Z'), (1, 'I'), (2, 'I'), (3, 'I')],
        [(0, 'I'), (1, 'Z'), (2, 'I'), (3, 'I')],
        [(0, 'I'), (1, 'I'), (2, 'Z'), (3, 'I')],
        [(0, 'I'), (1, 'I'), (2, 'I'), (3, 'Z')],
        [(0, 'Z'), (1, 'Z'), (2, 'I'), (3, 'I')],
        [(0, 'Y'), (1, 'X'), (2, 'X'), (3, 'Y')],
        [(0, 'Y'), (1, 'Y'), (2, 'X'), (3, 'X')],
        [(0, 'X'), (1, 'X'), (2, 'Y'), (3, 'Y')],
        [(0, 'X'), (1, 'Y'), (2, 'Y'), (3, 'X')],
        [(0, 'Z'), (1, 'I'), (2, 'Z'), (3, 'I')],
        [(0, 'Z'), (1, 'I'), (2, 'I'), (3, 'Z')],
        [(0, 'I'), (1, 'Z'), (2, 'Z'), (3, 'I')],
        [(0, 'I'), (1, 'Z'), (2, 'I'), (3, 'Z')],
        [(0, 'I'), (1, 'I'), (2, 'Z'), (3, 'Z')]
    ]


    :param constants:
    :type constants: list
    e.g.
        [
            (-0.32760818995565577+0j),
            (0.1371657293179602+0j),
            (0.1371657293179602+0j),
            (-0.13036292044009176+0j),
            (-0.13036292044009176+0j),
            (0.15660062486143395+0j),
            (0.04919764587885283+0j),
            (-0.04919764587885283+0j),
            (-0.04919764587885283+0j),
            (0.04919764587885283+0j),
            (0.10622904488350779+0j),
            (0.15542669076236065+0j),
            (0.15542669076236065+0j),
            (0.10622904488350779+0j),
            (0.1632676867167479+0j)
        ]

    :return:
    :rtype: list
    e.g.
        [
            ('I0 I1 I2 I3', (-0.32760818995565577+0j)),
            ('Z0 I1 I2 I3', (0.1371657293179602+0j)),
            ('I0 Z1 I2 I3', (0.1371657293179602+0j)),
            ('I0 I1 Z2 I3', (-0.13036292044009176+0j)),
            ('I0 I1 I2 Z3', (-0.13036292044009176+0j)),
            ('Z0 Z1 I2 I3', (0.15660062486143395+0j)),
            ('Y0 X1 X2 Y3', (0.04919764587885283+0j)),
            ('Y0 Y1 X2 X3', (-0.04919764587885283+0j)),
            ('X0 X1 Y2 Y3', (-0.04919764587885283+0j)),
            ('X0 Y1 Y2 X3', (0.04919764587885283+0j)),
            ('Z0 I1 Z2 I3', (0.10622904488350779+0j)),
            ('Z0 I1 I2 Z3', (0.15542669076236065+0j)),
            ('I0 Z1 Z2 I3', (0.15542669076236065+0j)),
            ('I0 Z1 I2 Z3', (0.10622904488350779+0j)),
            ('I0 I1 Z2 Z3', (0.1632676867167479+0j))
         ]

    """
    PauliWords_and_constants = []
    for i in range(len(PauliWords)):
        PauliWord = PauliWords[i]
        constant = constants[i]
        pauliword_string=[]
        for qubitNo, qubitOp in PauliWord:
            pauliword_string.append('{}{}'.format(qubitOp, qubitNo))
        seperator = ' '
        PauliWords_and_constants.append((seperator.join(pauliword_string), constant))
    return PauliWords_and_constants

def Get_quantum_circuits_and_constants_NORMAL(full_anstaz_circuit, PauliWords_and_constants):

    circuits_and_constants={}
    for key in range(len(PauliWords_and_constants)):

        PauliWord_constant = PauliWords_and_constants[key]

        Pauli_circuit_object = Perform_PauliWord_and_Measure(PauliWord_constant)
        q_circuit_Pauliword = cirq.Circuit.from_ops(
            cirq.decompose_once(
                (Pauli_circuit_object(*cirq.LineQubit.range(Pauli_circuit_object.num_qubits())))))
        circuit_ops = list(q_circuit_Pauliword.all_operations())

        if circuit_ops == []:
            # deals with identity only circuit
            circuits_and_constants[key] = {'circuit': None,
                                           'gamma_l': PauliWord_constant[1], 'PauliWord': PauliWord_constant[0]}
        else:
            full_circuit = cirq.Circuit.from_ops(
                [
                    *full_anstaz_circuit.all_operations(), # maybe make this a variable! (rather than repeated method)
                    *circuit_ops
                ])

            circuits_and_constants[key] = {'circuit': full_circuit,
                                           'gamma_l': PauliWord_constant[1], 'PauliWord': PauliWord_constant[0]}
    return circuits_and_constants



from scipy.optimize import minimize
import matplotlib.pyplot as plt
class OptimizerSTANDARD:
    '''
    Base class for optimizers. To specify a new optimization technique simply define a new objective function
    '''

    def __init__(self, num_shots, theta_guess_list, HF_state_prep_circuit, HF_initial_state, PauliWords_and_constants,
                 # All_X_sk_terms,
                 noisy=True, store_values=False, optimized_result=None):

        self.num_shots = num_shots
        self.initial_guess = theta_guess_list
        self.HF_state_prep_circuit = HF_state_prep_circuit
        # self.All_X_sk_terms = All_X_sk_terms
        self.HF_initial_state = HF_initial_state
        self.PauliWords_and_constants = PauliWords_and_constants

        self.iters = 0
        self.reps = num_shots
        self.obj_fun_values = []
        self.noisy = noisy
        self.store_values = store_values
        self.optimized_result = optimized_result
        self.theta_params = None

    def set_noise(self, _bool):
        self.noisy = _bool
        self.store_values = _bool

    def set_theta_params(self, params):
        # params will be numpy array of
        self.theta_params = params

    def set_reps(self, reps):
        self.reps = reps

    def objective_function(self, param_obj_fun):
        # Is full Objective function, with all parameters
        """"
        Returns Energy value... to be minimized!

         """
        UCC = Full_state_prep_circuit(self.HF_initial_state, T1_and_T2_theta_list=param_obj_fun)
        UCC.complete_UCC_circuit()
        UCC_quantum_circuit = UCC.UCC_full_circuit

        full_anstaz_circuit = cirq.Circuit.from_ops(
            [
                cirq.decompose_once(self.HF_state_prep_circuit),
                cirq.decompose_once(UCC_quantum_circuit)
            ]
        )

        quantum_circuit_dict = Get_quantum_circuits_and_constants_NORMAL(full_anstaz_circuit,
                                                                         self.PauliWords_and_constants)

        # quantum_circuit_dict = Get_quantum_circuits_and_constants(self.All_X_sk_terms, full_anstaz_circuit)

        sim = Simulation_Quantum_Circuit_Dict(quantum_circuit_dict, self.num_shots)
        Energy = sim.Calc_energy_via_parity()

        return Energy

    def callback_store_values(self, xk):
        val = self.objective_function(xk)
        self.obj_fun_values.append(val)
        if self.noisy:
            print(f'{self.iters}: angles: {xk}: Energy{val}')  # self.iters and xk
        self.iters += 1

    #    def AngleBounds(self):
    #        b = (0, 2*math.pi)
    #        bnds = [b for i in range(len(self.initial_guess))]

    def get_env(self, max_iter):
        options = {'maxiter': max_iter,
                   'disp': self.noisy}  # if noisy else False}

        kwargs = {'fun': self.objective_function,
                  'x0': self.initial_guess,  # = param_obj_fun
                  'method': 'Nelder-Mead',
                  'tol': 1e-5,
                  'options': options,
                  'callback': self.callback_store_values if self.store_values else None}

        self.optimized_result = minimize(**kwargs)  # scipy.optimize.minimize
        if self.noisy:
            print(f'Reason for termination is {self.optimized_result.message}')

    def plot_convergence(self):  # , file):
        # dir_path = os.path.dirname(os.path.realpath(__file__))
        plt.figure()
        x = list(range(len(self.obj_fun_values)))
        plt.plot(x, self.obj_fun_values)
        plt.xlabel('iterations')
        plt.ylabel('objective function value')
        # plt.savefig(dir_path + '/' + file)