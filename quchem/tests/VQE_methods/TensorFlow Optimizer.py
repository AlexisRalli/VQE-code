import matplotlib.pyplot as plt
import tensorflow as tf
import random
import numpy as np

if __name__ == '__main__':
    from Ansatz_Generator_Functions import *
    from Simulating_Quantum_Circuit import *
    from Unitary_partitioning import *
else:
    from .Ansatz_Generator_Functions import *
    from .Simulating_Quantum_Circuit import *
    from .Unitary_partitioning import *



def Wrapper_Objective_Function(anotherfunc, extraArgs):
    anotherfunc(*extraArgs)





def Cacluate_Energy():
    HF_UCC = Full_state_prep_circuit(HF_initial_state, T1_and_T2_theta_list=param_obj_fun)
    HF_UCC.complete_UCC_circuit()
    full_anstaz_circuit = HF_UCC.UCC_full_circuit

    UnitaryPart = UnitaryPartition(anti_commuting_sets, full_anstaz_circuit, S=0)
    UnitaryPart.Get_Quantum_circuits_and_constants()
    quantum_circuit_dict = UnitaryPart.circuits_and_constants

    sim = Simulation_Quantum_Circuit_Dict(quantum_circuit_dict, num_shots)
    Energy = sim.Calc_energy_via_parity()

    return Energy



class TensorFlowOptimizer:
    '''
    Base class for optimizers. To specify a new optimization technique simply define a new objective function
    '''

    def __init__(self, num_shots, theta_guess_list,
                 HF_initial_state, T1_formatted, T2_formatted, anti_commuting_sets,
                 learning_rate=0.001,
                 noisy=True, store_values=False, optimized_result=None,
                 optimizer = 'Adam', beta1=0.9, beta2=0.999):

        self.num_shots = num_shots
        self.initial_guess = theta_guess_list

        self.HF_initial_state = HF_initial_state
        self.T1_formatted = T1_formatted
        self.T2_formatted = T2_formatted
        self.anti_commuting_sets = anti_commuting_sets



        self.iters = 0
        self.reps = num_shots
        self.obj_fun_values = []
        self.noisy = noisy
        self.store_values = store_values
        self.optimized_result = optimized_result
        self.theta_params = None
        self.num_shots = num_shots

        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.beta1 = beta1
        self.beta2 = beta2

        tf.reset_default_graph()  # Resets  graph... IMPORTANT

    def set_noise(self, _bool):
        self.noisy = _bool
        self.store_values = _bool

    def Get_T1_T2_theta_list_tensors(self, T1_and_T2_theta_list):
        """
        Setup theta guess list as tensors

        Args:
            theta_guess_list = list of theta paramters
                               e.g. [1, 2, 4]
        Returns:
            A list of Tensors, corresponding to theta_guess_list
        """



        if T1_and_T2_theta_list == []:
            T1_tensor_list = [tf.Variable(random.uniform(0, 2 * math.pi), dtype=tf.float32) for i in range(len(self.T1_formatted))]
            T2_tensor_list = [tf.Variable(random.uniform(0, 2 * math.pi), dtype=tf.float32) for i in range(len(self.T2_formatted))]

        else:
            length_T1 = len(self.T1_formatted)
            length_T2 = len(self.T2_formatted)

            if len(T1_and_T2_theta_list) != length_T1 + length_T2:
                raise ValueError('Not enough angles defined. Have {} instead of {} angles.'
                                 'ALTERNATIVELY one can use an empty list to generate random angles.'.format(
                    len(T1_and_T2_theta_list), (length_T1 + length_T2)))
            T1_tensor_list = [tf.Variable(T1_and_T2_theta_list[i], dtype=tf.float32) for i in range(length_T1)]
            T2_tensor_list = [tf.Variable(T1_and_T2_theta_list[i + length_T1], dtype=tf.float32) for i in range(length_T2)]

        return T1_tensor_list + T2_tensor_list



    def Get_Energy(self, tensor_guess_list):
        # Is full Objective function, with all parameters
        """"
        Returns Energy value... to be minimized!

         Args:
             theta_guess_list: A list of theta parameters

             num_shots: number of repeats in simulation

             Hamiltonian_ConstantList: factor to times expectation value by

             IdentityConstant: Constant term to add

             Combined_T_parameters: A dictionary containing all circuit symbols and associated values
                                     e.g. {'T1_20': 0, 'T1_31': 1, 'T2_3210': 2}

             Returns:
                 Energy value
         """
        sess = tf.Session()
        init = tf.global_variables_initializer()

        sess.run(init)
        HF_UCC = Full_state_prep_circuit(self.HF_initial_state, T1_and_T2_theta_list=tensor_guess_list)
        HF_UCC.complete_UCC_circuit()
        full_anstaz_circuit = HF_UCC.UCC_full_circuit
        UnitaryPart = UnitaryPartition(anti_commuting_sets, full_anstaz_circuit, S=0)
        UnitaryPart.Get_Quantum_circuits_and_constants()
        quantum_circuit_dict = UnitaryPart.circuits_and_constants

        sim = Simulation_Quantum_Circuit_Dict(quantum_circuit_dict, self.num_shots)
        Energy = sim.Calc_energy_via_parity()

        self.E = tf.Variable(Energy, name='energy', dtype=tf.float32)

        return self.E




    def Gradient_angles_setup(self, T1_and_T2_theta_list):
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

          [ {'T1_20': a +pi/4,   'T1_31': b,   T2_3210': c}
            {'T1_20': a,   'T1_31': b +pi/4,   T2_3210': c}
            {'T1_20': a,   'T1_31': b,   T2_3210': c +pi/4}
          ]

          and

          [ {'T1_20': a -pi/4,   'T1_31': b,   T2_3210': c}
            {'T1_20': a,   'T1_31': b -pi/4,   T2_3210': c}
            {'T1_20': a,   'T1_31': b,   T2_3210': c -pi/4}
          ]

        """

        Plus_parameter_list = []
        Minus_parameter_list = []

        for theta in T1_and_T2_theta_list:
            Plus_parameter_list.update(theta + (np.pi / 4))
            Minus_parameter_list.update(theta - (np.pi / 4))

        Plus_parameter_list = self.Get_T1_T2_theta_list_tensors(Plus_parameter_list)
        Minus_parameter_list = self.Get_T1_T2_theta_list_tensors(Minus_parameter_list)

        return Plus_parameter_list, Minus_parameter_list




    def Analytic_Gradient(self, tensor_guess_list):

       PLUS_tensors, MINUS_tensors = self.Gradient_angles_setup(tensor_guess_list)

       partial_gradient_list = []

       for j in range(len(tensor_guess_list)):

           tensor_guess_list_PLUS = tensor_guess_list
           tensor_guess_list_PLUS[j] =  PLUS_tensors[j]

           HF_UCC = Full_state_prep_circuit(self.HF_initial_state, T1_and_T2_theta_list=tensor_guess_list_PLUS)
           HF_UCC.complete_UCC_circuit()
           full_anstaz_circuit = HF_UCC.UCC_full_circuit
           UnitaryPart = UnitaryPartition(self.anti_commuting_sets, full_anstaz_circuit, S=0)
           UnitaryPart.Get_Quantum_circuits_and_constants()
           quantum_circuit_dict = UnitaryPart.circuits_and_constants

           sim = Simulation_Quantum_Circuit_Dict(quantum_circuit_dict, self.num_shots)
           HAM_PLUS = sim.Calc_energy_via_parity()


           tensor_guess_list_MINUS = tensor_guess_list
           tensor_guess_list_MINUS[j] =  MINUS_tensors[j]

           HF_UCC = Full_state_prep_circuit(self.HF_initial_state, T1_and_T2_theta_list=tensor_guess_list_MINUS)
           HF_UCC.complete_UCC_circuit()
           full_anstaz_circuit = HF_UCC.UCC_full_circuit
           UnitaryPart = UnitaryPartition(self.anti_commuting_sets, full_anstaz_circuit, S=0)
           UnitaryPart.Get_Quantum_circuits_and_constants()
           quantum_circuit_dict = UnitaryPart.circuits_and_constants

           sim = Simulation_Quantum_Circuit_Dict(quantum_circuit_dict, self.num_shots)
           Ham_MINUS = sim.Calc_energy_via_parity()

           Gradient = (HAM_PLUS - Ham_MINUS)  # /2
           partial_gradient_list.append((Gradient, tensor_guess_list[j]))

       return partial_gradient_list




    def Feed_Dictrionary(self, gradient_sub, gradient_LIST):
        feed_dictionary = {}
        i = 0
        for tensor_var in gradient_sub:
            while i < len(gradient_LIST):
                dic = {tensor_var: gradient_LIST[i][0]}  # {tensor: value_of_tensor}
                feed_dictionary.update(dic)
                i += 1
                break
        return feed_dictionary

    def Optimize(self, max_iter):
        tensor_guess_list = self.GetTensorFlowTensors()

        ENERGY_VAL = self.Get_Energy(tensor_guess_list)
        sess = tf.Session()
        init = tf.global_variables_initializer()  # for some reason i need to repeat this!
        sess.run(init)

        if self.optimizer == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1, beta2=self.beta2)
        elif self.optimizer == 'GradientDescent':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        else:
            raise print('Optimizer Not defined')

        grad_PH_and_label = self.GradientSubPlaceHolderList()
        grad_place_holder_list = [grad_PH_and_label[key] for key in grad_PH_and_label]
        gradient_sub = (*grad_place_holder_list, 0.0)  # extra 0.0 due to varibles having one extra value!

        _, variables = zip(*optimizer.compute_gradients(ENERGY_VAL))

        train_step = optimizer.apply_gradients(zip(gradient_sub, variables))
        init = tf.global_variables_initializer()

        angle_list = []

        with tf.Session() as session:
            session.run(init)
            print("starting at", "Angles:", session.run(tensor_guess_list))
            for step in range(max_iter):
                new_angles = session.run(tensor_guess_list)
                gradient_LIST = self.Analytic_Gradient(new_angles)
                print('Gradients: ', gradient_LIST)
                feed_dictionary = self.Feed_Dictrionary(gradient_sub, gradient_LIST)
                train_step.run(feed_dict=feed_dictionary)
                print("step", step, "angles:",
                      new_angles)  # , "Energy:", session.run(Energy(tensor_guess_list, Combined_T_parameters)))

                print('Energy:', session.run(self.E))
                print("")
                self.iters += 1
                self.obj_fun_values.append(session.run(tensor_guess_list))

                angle_list.append(THETA_guess_list)

        index = self.obj_fun_values.index(min(self.obj_fun_values))
        self.optimized_result = {'Angles': angle_list[index], 'Energy': min(self.obj_fun_values)}







        ####
        tensor_guess_list = self.GetTensorFlowTensors()

        ENERGY_VAL = self.Get_Energy(tensor_guess_list)
        sess = tf.Session()
        # init = tf.global_variables_initializer()  # for some reason i need to repeat this!
        # sess.run(init)


        if self.optimizer == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1, beta2=self.beta2)
        elif self.optimizer == 'GradientDescent':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        else:
            raise print('Optimizer Not defined')

        gradient_sub = self.Analytic_Gradient(tensor_guess_list)


        _, variables = zip(*optimizer.compute_gradients(ENERGY_VAL))

        train_step = optimizer.apply_gradients(zip(gradient_sub, variables))
        init = tf.global_variables_initializer()
        sess.run(init)

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
                print('Energy:', session.run(self.E))
                print("")
                self.iters +=1
                self.obj_fun_values.append(self.Energy_PRINT(THETA_guess_list))

                angle_list.append(THETA_guess_list)

        index = self.obj_fun_values.index(min(self.obj_fun_values))
        self.optimized_result = {'Angles': angle_list[index], 'Energy': min(self.obj_fun_values)}

    def set_reps(self, reps):
        self.reps = reps

    def callback_store_values(self, xk):
        val = self.objective_function(xk)
        self.obj_fun_values.append(val)
        if self.noisy:
            print(f'{self.iters}: angles: {xk}: Energy{val}')  # self.iters and xk
        self.iters += 1

    def plot_convergence(self):  # , file):
        # dir_path = os.path.dirname(os.path.realpath(__file__))
        plt.figure()
        x = list(range(len(self.obj_fun_values)))
        plt.plot(x, self.obj_fun_values)
        plt.xlabel('iterations')
        plt.ylabel('objective function value')
        plt.show()
        # plt.savefig(dir_path + '/' + file)




# x = TensorFlowOptimizer(num_shots, theta_guess_list,
#                  Hamiltonian_ConstantList, IdentityConstant,
#                  Combined_T_parameters, Full_instance_Circuit_Gen_List,
#                  Hamiltonian_QubitNoList, Hamiltonian_OperationList, 0.001,
#                  noisy=True, store_values=False, optimized_result=None,
#                  optimizer = 'Adam', beta1=0.9, beta2=0.999)
#
# x.Optimize(10)