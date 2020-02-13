from quchem.Simulating_Quantum_Circuit import *
from quchem.Ansatz_Generator_Functions import *


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
    ansatz_circ = list(full_anstaz_circuit.all_operations())
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
                    *ansatz_circ,
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

    def __init__(self, num_shots, theta_guess_list, HF_initial_state, PauliWords_and_constants,
                 # All_X_sk_terms,
                 noisy=True, store_values=False, optimized_result=None):

        self.num_shots = num_shots
        self.initial_guess = theta_guess_list
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
        HF_UCC = Full_state_prep_circuit(self.HF_initial_state, T1_and_T2_theta_list=param_obj_fun)
        HF_UCC.complete_UCC_circuit()
        full_anstaz_circuit = HF_UCC.UCC_full_circuit

        quantum_circuit_dict = Get_quantum_circuits_and_constants_NORMAL(full_anstaz_circuit,
                                                                         self.PauliWords_and_constants)


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

import tensorflow as tf

class TensorFlow_Optimizer_STANDARD():

    def __init__(self, initial_theta_guess_LIST, HF_initial_state, num_shots, PauliWords_and_constants,
                 learning_rate=0.01,
                 optimizer = 'Adam', beta1=0.9, beta2=0.999):

        self.initial_theta_guess_LIST = initial_theta_guess_LIST
        self.HF_initial_state = HF_initial_state
        self.PauliWords_and_constants =  PauliWords_and_constants
        self.num_shots = num_shots
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        tf.reset_default_graph()

    def Calc_Energy_NORMAL(self, theta_list):
        HF_UCC = Full_state_prep_circuit(self.HF_initial_state, T1_and_T2_theta_list=theta_list)
        HF_UCC.complete_UCC_circuit()
        full_anstaz_circuit = HF_UCC.UCC_full_circuit

        quantum_circuit_dict = Get_quantum_circuits_and_constants_NORMAL(full_anstaz_circuit,
                                                                         self.PauliWords_and_constants)

        sim = Simulation_Quantum_Circuit_Dict(quantum_circuit_dict, self.num_shots)
        Energy = sim.Calc_energy_via_parity()
        Energy = np.float32(Energy.real)
        return Energy

    def Gradient_angles_setup(self, theta_list):
        """
        Args:
            theta_guess_list = List of T1 guesses followed by T2 guesses
                                e.g. [a,b,c] (e.g. [1,0,3])

            circuit_label_dictionary = Dict of symbols and values
                        e.g.{'T1_20': a,   'T1_31': b,   T2_3210': c}

        Returns:
            List of theta parameters with theta value +pi/4 and -pi/4 for each item in list.
          e.g:

         [ 'T1_20': a + pi/4,   'T1_31': b + pi/4,   T2_3210': c + pi/4 ]
          and
          [ 'T1_20': a - pi/4,   'T1_31': b - pi/4,   T2_3210': c - pi/4 ]

        """

        Plus_parameter_list = []
        Minus_parameter_list = []

        for theta in theta_list:
            Plus_parameter_list.append((theta + np.pi / 4))
            Minus_parameter_list.append((theta - np.pi / 4))

        return Plus_parameter_list, Minus_parameter_list

    def Gradient_funct_NORMAL(self, theta_guess_list):

        theta_list_PLUS_full, theta_list_MINUS_full = self.Gradient_angles_setup(theta_guess_list)

        partial_gradient_list = []
        for j in range(len(theta_guess_list)):
            theta = theta_guess_list[j]

            theta_list_PLUS = theta_guess_list
            theta_list_PLUS[j] = theta_list_PLUS_full[j]
            Ham_PLUS = self.Calc_Energy_NORMAL(theta_list_PLUS)

            theta_list_MINUS = theta_guess_list
            theta_list_MINUS[j] = theta_list_MINUS_full[j]
            Ham_MINUS = self.Calc_Energy_NORMAL(theta_list_MINUS)

            Gradient = (Ham_PLUS - Ham_MINUS)  # /2
            partial_gradient_list.append((Gradient, theta)) #.append(Gradient)
        return partial_gradient_list

    def Calc_Energy_TENSOR(self, theta_list_TENSOR):

        sess = tf.Session()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        theta_list = [sess.run(theta) for theta in theta_list_TENSOR]

        output = self.Calc_Energy_NORMAL(theta_list)
        return tf.Variable(output, dtype=tf.float64)


    def Get_Gradient_PLACEHOLDERS(self):

        place_holder_list = [tf.placeholder(tf.float32, shape=(), name="{}".format(i)) for i in self.initial_theta_guess_LIST]

        return place_holder_list


    def Feed_dictionary(self, placeholder_list, grads_and_vars):

        feed_dict = {}
        for i in range(len(placeholder_list)):
            PH = placeholder_list[i]
            gradient = grads_and_vars[i][0]
            feed_dict.update({PH: gradient})
        return feed_dict


    def optimize(self, max_iter):

        theta_guess = self.initial_theta_guess_LIST
        theta_list_TENSOR = [tf.Variable(i, dtype=tf.float32) for i in theta_guess]



        if self.optimizer == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1, beta2=self.beta2)
        elif self.optimizer == 'GradientDescent':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        else:
            raise ValueError('Optimizer Not defined')

        init = tf.compat.v1.global_variables_initializer()

        Energy = self.Calc_Energy_TENSOR(theta_list_TENSOR)
        _, variables = zip(*optimizer.compute_gradients(Energy))

        place_holder_list = self.Get_Gradient_PLACEHOLDERS()

        train_step = optimizer.apply_gradients(zip(place_holder_list, variables))


        self.E_list =[]
        self.Angle_list =[]

        with tf.Session() as session:
            session.run(init)
            Angles = [session.run(theta_tensor) for theta_tensor in theta_list_TENSOR]
            Energy = self.Calc_Energy_NORMAL(Angles)
            print("starting at Angles:", Angles, "Energy:",
                  Energy)

            self.Angle_list.append(Angles)
            self.E_list.append(Energy)

            for step in range(max_iter):
                if step == 0:
                    session.run(tf.compat.v1.global_variables_initializer())

                    Angles = [session.run(theta_tensor) for theta_tensor in theta_list_TENSOR]
                    grads_and_vars = self.Gradient_funct_NORMAL(Angles)
                    grad_sub_dict = self.Feed_dictionary(place_holder_list, grads_and_vars)

                    train_step.run(feed_dict=grad_sub_dict)

                    NEW_Angles = [session.run(theta_tensor) for theta_tensor in theta_list_TENSOR]
                    Energy = self.Calc_Energy_NORMAL(NEW_Angles)
                    print("step", step, "Angles:", NEW_Angles, "Energy:", Energy)

                    self.Angle_list.append(NEW_Angles)
                    self.E_list.append(Energy)

                else:
                    Angles = [session.run(theta_tensor) for theta_tensor in theta_list_TENSOR]
                    grads_and_vars = self.Gradient_funct_NORMAL(Angles)

                    grad_sub_dict = self.Feed_dictionary(place_holder_list, grads_and_vars)

                    train_step.run(feed_dict=grad_sub_dict)

                    NEW_Angles = [session.run(theta_tensor) for theta_tensor in theta_list_TENSOR]
                    Energy = self.Calc_Energy_NORMAL(NEW_Angles)
                    print("step", step, "Angles:", NEW_Angles, "Energy:", Energy)

                    self.Angle_list.append(NEW_Angles)
                    self.E_list.append(Energy)


    def plot_convergence(self):  # , file):
        # dir_path = os.path.dirname(os.path.realpath(__file__))
        plt.figure()
        x = list(range(len(self.E_list)))
        plt.plot(x, self.E_list)
        plt.xlabel('iterations')
        plt.ylabel('objective function value')
        # plt.savefig(dir_path + '/' + file)