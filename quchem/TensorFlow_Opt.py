import matplotlib.pyplot as plt
import tensorflow as tf


from quchem.Ansatz_Generator_Functions import *
from quchem.Simulating_Quantum_Circuit import *
from quchem.Unitary_partitioning import *



class TensorFlow_Optimizer():

    def __init__(self, initial_theta_guess_LIST, HF_initial_state, num_shots,
                 learning_rate=0.01,
                 optimizer = 'Adam', beta1=0.9, beta2=0.999):

        self.initial_theta_guess_LIST = initial_theta_guess_LIST
        self.HF_initial_state = HF_initial_state
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

        UnitaryPart = UnitaryPartition(anti_commuting_sets, full_anstaz_circuit, S=0)
        UnitaryPart.Get_Quantum_circuits_and_constants()
        quantum_circuit_dict = UnitaryPart.circuits_and_constants

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

    # def Gradient_funct_TENSOR(self, theta_list_TENSOR):
    #     sess = tf.Session()
    #     init = tf.compat.v1.global_variables_initializer()
    #     sess.run(init)
    #
    #     theta_list = [sess.run(theta) for theta in theta_list_TENSOR]
    #
    #     gradient_list = self.Gradient_funct_NORMAL(theta_list)
    #     return zip(gradient_list, theta_list_TENSOR)

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

        # grads_and_vars = self.Gradient_funct_TENSOR(theta_list_TENSOR)
        # optimizer.apply_gradients(grads_and_vars)
        #
        # function = self.Calc_Energy_TENSOR(theta_list_TENSOR)
        #
        # init = tf.compat.v1.global_variables_initializer()
        # self.E_list =[]
        # self.Angle_list =[]
        #
        # with tf.compat.v1.Session() as session:
        #     session.run(init)
        #     Angles = [session.run(var) for var in theta_list_TENSOR]
        #     Energy = session.run(function)
        #     print("starting at variables:", Angles, "Energy:", Energy)
        #     self.E_list.append(Energy)
        #     self.Angle_list.append(Angles)
        #
        #     for step in range(max_iter):
        #         grads_and_vars = self.Gradient_funct_TENSOR(theta_list_TENSOR)
        #         train = optimizer.apply_gradients(grads_and_vars)
        #         # init = tf.compat.v1.global_variables_initializer()    <- seems WRONG
        #         # session.run(init)                                     <- seems WRONG
        #
        #         session.run(train)
        #
        #         Angles = [session.run(angle) for angle in theta_list_TENSOR]
        #         Energy = self.Calc_Energy_NORMAL(Angles)  # NOTE this is not the tensor function
        #         # Energy = session.run(function) <--- note it is NOT THIS!
        #         print("step", step, "variables:", Angles, "Energy:", Energy)
        #         self.E_list.append(Energy)
        #         self.Angle_list.append(Angles)





    def plot_convergence(self):  # , file):
        # dir_path = os.path.dirname(os.path.realpath(__file__))
        plt.figure()
        x = list(range(len(self.E_list)))
        plt.plot(x, self.E_list)
        plt.xlabel('iterations')
        plt.ylabel('objective function value')
        # plt.savefig(dir_path + '/' + file)





# theta_guess = [random.uniform(0, 2*math.pi) for i in range(3)]
# theta_guess = [1.7710197, 4.8140006, 0.316327]

# xx = TensorFlow_Optimizer(theta_guess, HF_initial_state, 10000,
#                  learning_rate=0.01,
#                  optimizer = 'Adam', beta1=0.9, beta2=0.999)
# xx.optimize(150)


