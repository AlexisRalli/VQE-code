import matplotlib.pyplot as plt
import tensorflow as tf


from quchem.Ansatz_Generator_Functions import *
from quchem.Simulating_Quantum_Circuit import *
from quchem.Unitary_partitioning import *


def Calc_Energy(theta_list, HF_initial_state, num_shots=10000):
    HF_UCC = Full_state_prep_circuit(HF_initial_state, T1_and_T2_theta_list=theta_list)
    HF_UCC.complete_UCC_circuit()
    full_anstaz_circuit = HF_UCC.UCC_full_circuit

    UnitaryPart = UnitaryPartition(anti_commuting_sets, full_anstaz_circuit, S=0)
    UnitaryPart.Get_Quantum_circuits_and_constants()
    quantum_circuit_dict = UnitaryPart.circuits_and_constants

    sim = Simulation_Quantum_Circuit_Dict(quantum_circuit_dict, num_shots)
    Energy = sim.Calc_energy_via_parity()
    return Energy

def Gradient_angles_setup(theta_list):
    """
    Args:
        theta_guess_list = List of T1 guesses followed by T2 guesses
                            e.g. [a,b,c] (e.g. [1,0,3])

        circuit_label_dictionary = Dict of symbols and values
                    e.g.{'T1_20': a,   'T1_31': b,   T2_3210': c}

    Returns:
        A list of cirq.ParamResolvers
        Each list gives the partial derivative for one variable!
        (aka one variable changed by pi/4 and -pi/4 the rest kept constant!)

        Note this is a cirq.ParamResolver
        note: dH(θ)/dθ = H(θ+ pi/4) - H(θ - pi/4)

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

def Gradient_funct(theta_guess_list, HF_initial_state):

    theta_list_PLUS_full, theta_list_MINUS_full = Gradient_angles_setup(theta_guess_list)

    partial_gradient_list = []
    for j in range(len(theta_guess_list)):

        theta = theta_guess_list[j]

        theta_list_PLUS = theta_guess_list
        theta_list_PLUS[j] = theta_list_PLUS_full[j]
        Ham_PLUS = Calc_Energy(theta_list_PLUS, HF_initial_state, num_shots = 10000)



        theta_list_MINUS = theta_guess_list
        theta_list_MINUS[j] = theta_list_MINUS_full[j]
        Ham_MINUS = Calc_Energy(theta_list_MINUS, HF_initial_state, num_shots = 10000)


        Gradient = (Ham_PLUS - Ham_MINUS)  # /2
        partial_gradient_list.append((Gradient, theta))
    return partial_gradient_list


class Standard_Energy_Calc():
    def __init__(self, HF_initial_state, num_shots):
        self.HF_initial_state = HF_initial_state
        self.num_shots = num_shots


    def Calculate_ENERGY(self, theta_Guess_List):
        E = Calc_Energy(theta_Guess_List, self.HF_initial_state, self.num_shots)
        return E

    def Calculate_Gradient(self, theta_Guess_List):
        grads_and_vars = Gradient_funct(theta_Guess_List, self.HF_initial_state)
        return grads_and_vars

x = Standard_Energy_Calc(HF_initial_state, 1000)
x.Calculate_ENERGY([0,1,2])
x.Calculate_Gradient([0,1,2])



class TensorFlow_OPT():

    def __init__(self,tensor_guess_list, HF_state, num_shots,
                 learning_rate=0.001,
                 optimizer = 'Adam', beta1=0.9, beta2=0.999):
        self.tensor_guess_list = tensor_guess_list
        self.HF_state = HF_state
        self.num_shots = num_shots

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        tf.reset_default_graph()

    def Get_E(self, tensor_guess_list):
        sess = tf.Session()
        init = tf.global_variables_initializer()  # for some reason i need to repeat this!
        sess.run(init)

        theta_list = [sess.run(i) for i in tensor_guess_list]
        Energy = Calc_Energy(theta_list, self.HF_state, num_shots= self.num_shots)
        return tf.Variable(Energy, name='energy', dtype=tf.float32)

    def Analytic_Gradient(self, theta_guess_list):
        return Gradient_funct(theta_guess_list, self.HF_state)

    def optimize(self, max_iter):
        ENERGY_VAL = self.Get_E(self.tensor_guess_list)

        print(ENERGY_VAL)

        sess = tf.Session()
        init = tf.global_variables_initializer()  # for some reason i need to repeat this!
        sess.run(init)

        if self.optimizer == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1, beta2=self.beta2)
        elif self.optimizer == 'GradientDescent':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        else:
            raise print('Optimizer Not defined')

        _, variables = zip(*optimizer.compute_gradients(ENERGY_VAL))

        train_step = optimizer.apply_gradients(zip(gradient_sub, variables))
        init = tf.global_variables_initializer()

        angle_list=[]

        with tf.Session() as session:
            session.run(init)
            print("starting at", "Angles:", session.run(tensor_guess_list))
            for step in range(max_iter):

                new_angles = session.run(tensor_guess_list)
                gradient_LIST = self.Analytic_Gradient(new_angles)
                print('Gradients: ', gradient_LIST)
                feed_dictionary = self.Feed_Dictrionary(gradient_sub, gradient_LIST)
                train_step.run(feed_dict=feed_dictionary)
                print("step", step, "angles:", new_angles)#, "Energy:", session.run(Energy(tensor_guess_list, Combined_T_parameters)))

                THETA_guess_list = session.run(tensor_guess_list)
                print('Energy:', self.Energy_PRINT(THETA_guess_list))
                print("")
                self.iters +=1
                self.obj_fun_values.append(self.Energy_PRINT(THETA_guess_list))

                angle_list.append(THETA_guess_list)





def Calc_Energy_TENSOR(theta_TENSOR_list, HF_initial_state, num_shots=10000):
    sess = tf.Session()
    init = tf.global_variables_initializer()  # for some reason i need to repeat this!
    sess.run(init)
    theta_list = [sess.run(i) for i in theta_TENSOR_list]
    Energy = Calc_Energy(theta_list, HF_initial_state, num_shots=num_shots)

    Energy = tf.Variable(Energy, name='energy', dtype=tf.float32)
    return Energy

def Gradient_angles_setup(theta_list):
    """
    Args:
        theta_guess_list = List of T1 guesses followed by T2 guesses
                            e.g. [a,b,c] (e.g. [1,0,3])

        circuit_label_dictionary = Dict of symbols and values
                    e.g.{'T1_20': a,   'T1_31': b,   T2_3210': c}

    Returns:
        A list of cirq.ParamResolvers
        Each list gives the partial derivative for one variable!
        (aka one variable changed by pi/4 and -pi/4 the rest kept constant!)

        Note this is a cirq.ParamResolver
        note: dH(θ)/dθ = H(θ+ pi/4) - H(θ - pi/4)

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

# def Gradient_funct_TENSOR(theta_TENSOR_list, HF_initial_state):
#
#     sess = tf.Session()
#     init = tf.global_variables_initializer()  # for some reason i need to repeat this!
#     sess.run(init)
#     theta_list = [sess.run(i) for i in theta_TENSOR_list]
#
#     theta_list_PLUS_full, theta_list_MINUS_full = Gradient_angles_setup(theta_list)
#
#     partial_gradient_list = []
#     for j in range(len(theta_list)):
#
#         theta_TENSOR = theta_TENSOR_list[j]
#
#         theta_list_PLUS = theta_list
#         theta_list_PLUS[j] = theta_list_PLUS_full[j]
#         Ham_PLUS = Calc_Energy(theta_list_PLUS, HF_initial_state, num_shots = 10000)
#
#
#
#         theta_list_MINUS = theta_list
#         theta_list_MINUS[j] = theta_list_MINUS_full[j]
#         Ham_MINUS = Calc_Energy(theta_list_MINUS, HF_initial_state, num_shots = 10000)
#
#
#         Gradient = (Ham_PLUS - Ham_MINUS)  # /2
#         partial_gradient_list.append((Gradient, theta_TENSOR))
#     return partial_gradient_list
def Gradient_funct_TENSOR_FEED_DICT(theta_TENSOR_list, HF_initial_state):
    feed_dictionary = {}

    sess = tf.Session()
    init = tf.global_variables_initializer()  # for some reason i need to repeat this!
    sess.run(init)
    theta_list = [sess.run(i) for i in theta_TENSOR_list]

    theta_list_PLUS_full, theta_list_MINUS_full = Gradient_angles_setup(theta_list)

    partial_gradient_list = []
    for j in range(len(theta_list)):

        theta_TENSOR = theta_TENSOR_list[j]

        theta_list_PLUS = theta_list
        theta_list_PLUS[j] = theta_list_PLUS_full[j]
        Ham_PLUS = Calc_Energy(theta_list_PLUS, HF_initial_state, num_shots = 10000)



        theta_list_MINUS = theta_list
        theta_list_MINUS[j] = theta_list_MINUS_full[j]
        Ham_MINUS = Calc_Energy(theta_list_MINUS, HF_initial_state, num_shots = 10000)


        Gradient = (Ham_PLUS - Ham_MINUS)  # /2
        partial_gradient_list.append((Gradient, theta_TENSOR))

        dic = {theta_TENSOR: Gradient}  # {tensor: value_of_tensor}
        feed_dictionary.update(dic)

    return feed_dictionary


def Optimize(max_iter, theta_list, HF_initial_state):
    tf.reset_default_graph()

    theta_TENSOR_list = [tf.Variable(var, dtype=tf.float32) for var in theta_list]

    ENERGY_VAL = Calc_Energy_TENSOR(theta_TENSOR_list, HF_initial_state)
    sess = tf.Session()
    init = tf.global_variables_initializer()  # for some reason i need to repeat this!
    sess.run(init)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)

    # place_holder = [tf.placeholder(tf.float32, shape=()) for i in theta_TENSOR_list]
    # gradient_sub = (*place_holder, 0.0)  # extra 0.0 due to varibles having one extra value!
    #
    # _, variables = zip(*optimizer.compute_gradients(ENERGY_VAL))
    # train_step = optimizer.apply_gradients(zip(gradient_sub, variables))

    feed_dictionary = Gradient_funct_TENSOR_FEED_DICT(theta_TENSOR_list, HF_initial_state)

    grads = ((tensor, partial_grad) for tensor, partial_grad in feed_dictionary.items())
    grads = (*grads, 0.0)  # extra 0.0 due to varibles having one extra value!

    train_step = optimizer.apply_gradients(grads)

    new_angles = sess.run(theta_TENSOR_list)
    print(new_angles)
    angle_list=[]

    with tf.Session() as session:
        session.run(init)
        print("starting at", "Angles:", session.run(theta_TENSOR_list))
        for step in range(max_iter):

            new_angles = session.run(theta_TENSOR_list)

            feed_dictionary = Gradient_funct_TENSOR_FEED_DICT(theta_TENSOR_list, HF_initial_state)
            print('Gradients: ', [partial_grad for tensor, partial_grad in feed_dictionary.items()])
            train_step.run(feed_dict=feed_dictionary)
            print("step", step, "angles:", new_angles)#, "Energy:", session.run(Energy(tensor_guess_list, Combined_T_parameters)))

            # THETA_guess_list = session.run(theta_TENSOR_list)
            # print('Energy:', self.Energy_PRINT(THETA_guess_list))
            # print("")
            # self.iters +=1
            # self.obj_fun_values.append(self.Energy_PRINT(THETA_guess_list))
            #
            # angle_list.append(THETA_guess_list)
