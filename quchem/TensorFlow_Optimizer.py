import tensorflow as tf
import matplotlib.pyplot as plt

class TensorFlow_Optimizer():

    def __init__(self,function_to_minimise, Gradient_function, kwargs,
                 learning_rate=0.001,
                 optimizer = 'Adam', beta1=0.9, beta2=0.999):
        self.function_to_minimise = function_to_minimise
        self.kwargs = kwargs
        self.Gradient_function = Gradient_function
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        tf.reset_default_graph()

    def Get_functions_args_as_tensors(self):
        constants=None
        for key in self.kwargs:
            if key == 'constants':
                constants = [tf.constant(const, dtype=tf.float32) for const in self.kwargs[key]]
            elif key == 'variables':
                vars = [tf.Variable(var, dtype=tf.float32) for var in self.kwargs[key]]

        if constants == None:
            return self.function_to_minimise(*vars), vars
        else:
            return self.function_to_minimise(*vars, *constants), vars


    def Calc_Gradient(self, vars):
        return self.Gradient_function(*vars)


    def optimize(self, max_iter):

        function, variables = self.Get_functions_args_as_tensors()
        grads_and_vars = self.Calc_Gradient(variables)


        if self.optimizer == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1, beta2=self.beta2)
        elif self.optimizer == 'GradientDescent':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        else:
            raise ValueError('Optimizer Not defined') #raise Error

        train = optimizer.apply_gradients(grads_and_vars)

        init = tf.compat.v1.global_variables_initializer()

        self.E_list =[]
        self.Angle_list =[]
        with tf.compat.v1.Session() as session:
            session.run(init)
            Angles = [session.run(var)for var in variables]
            Energy = session.run(function)
            print("starting at angles:", Angles,  "Energy:", Energy)
            self.E_list.append(Energy)
            self.Angle_list.append(Angles)

            for step in range(max_iter):

                session.run(train)

                Angles = [session.run(var) for var in variables]
                Energy = session.run(function)
                print("step", step, "Angles:", Angles, "Energy:", Energy)
                self.E_list.append(Energy)
                self.Angle_list.append(Angles)


    def plot_convergence(self):  # , file):
        # dir_path = os.path.dirname(os.path.realpath(__file__))
        plt.figure()
        x = list(range(len(self.E_list)))
        plt.plot(x, self.E_list)
        plt.xlabel('iterations')
        plt.ylabel('objective function value')
        # plt.savefig(dir_path + '/' + file)


if __name__ == '__main__':
    def function_to_minimize(x,y, const):
        # z = x^2 + y^2 + constant
        z = x**2 + y**2 + const
        return z

    def calc_grad(x,y):
        # z = 2x^2 + y^2 + constant
        dz_dx = 2*x
        dz_dy = 2*y
        return [(dz_dx, x), (dz_dy, y)]

    input_dict = {'variables': [2,3], 'constants': [2]}

    test = TensorFlow_Optimizer(function_to_minimize, calc_grad, input_dict,
                                optimizer='GradientDescent',  learning_rate=0.55)

    # test = TensorFlow_Optimizer(function_to_minimize, calc_grad, input_dict,
    #                             optimizer='Adam')

    test.optimize(20)

    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    x = np.arange(-10, 10, 0.25)
    y = np.arange(-10, 10, 0.25)
    const = 2

    x, y = np.meshgrid(x, y)
    z = x ** 2 + y ** 2 + const

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.viridis)
    plt.show()
    print('Minimum should be:', 2.0)


from quchem.Ansatz_Generator_Functions import *
from quchem.Simulating_Quantum_Circuit import *
from quchem.Unitary_partitioning import *


def Energy_obj_Funct(theta_list, HF_initial_state, num_shots=10000):
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
        Ham_PLUS = Energy_obj_Funct(theta_list_PLUS, HF_initial_state, num_shots = 10000)


        theta_list_MINUS = theta_guess_list
        theta_list_MINUS[j] = theta_list_MINUS_full[j]
        Ham_MINUS = Energy_obj_Funct(theta_list_MINUS, HF_initial_state, num_shots = 10000)


        Gradient = (Ham_PLUS - Ham_MINUS)  # /2
        partial_gradient_list.append((Gradient, theta))
    return partial_gradient_list


# ### note
# 1. run project_measurement)reduction_scipt
# 2. try running the above functions!
# 3 e.g. Energy_obj_Funct(0,1,2)





class TensorFlow_Optimizer():

    def __init__(self,function_to_minimise, Gradient_function, kwargs,
                 learning_rate=0.001,
                 optimizer = 'Adam', beta1=0.9, beta2=0.999):
        self.function_to_minimise = function_to_minimise
        self.kwargs = kwargs
        self.Gradient_function = Gradient_function
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2

    def Get_functions_args_as_tensors(self):
        constants=None
        for key in self.kwargs:
            if key == 'constants':
                constants = [tf.constant(const, dtype=tf.float32) for const in self.kwargs[key]]
            elif key == 'variables':
                vars = [tf.Variable(var, dtype=tf.float32) for var in self.kwargs[key]]

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        if constants == None:
            return self.function_to_minimise(*[sess.run(i) for i in vars]), vars
        else:
            return self.function_to_minimise(*[sess.run(i) for i in vars], *[sess.run(j) for j in constants]), vars


    def Calc_Gradient(self, vars):
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        return self.Gradient_function(*[sess.run(i) for i in vars])
        # return self.Gradient_function(vars)


    def optimize(self, max_iter):

        function, variables = self.Get_functions_args_as_tensors()
        grads_and_vars = self.Calc_Gradient(variables)


        if self.optimizer == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1, beta2=self.beta2)
        elif self.optimizer == 'GradientDescent':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        else:
            raise ValueError('Optimizer Not defined') #raise Error

        train = optimizer.apply_gradients(grads_and_vars)

        init = tf.compat.v1.global_variables_initializer()

        self.E_list =[]
        self.Angle_list =[]
        with tf.compat.v1.Session() as session:
            session.run(init)
            Angles = [session.run(var)for var in variables]
            Energy = session.run(function)
            print("starting at angles:", Angles,  "Energy:", Energy)
            self.E_list.append(Energy)
            self.Angle_list.append(Angles)

            for step in range(max_iter):

                session.run(train)

                Angles = [session.run(var) for var in variables]
                Energy = session.run(function)
                print("step", step, "Angles:", Angles, "Energy:", Energy)
                self.E_list.append(Energy)
                self.Angle_list.append(Angles)


    def plot_convergence(self):  # , file):
        # dir_path = os.path.dirname(os.path.realpath(__file__))
        plt.figure()
        x = list(range(len(self.E_list)))
        plt.plot(x, self.E_list)
        plt.xlabel('iterations')
        plt.ylabel('objective function value')
        # plt.savefig(dir_path + '/' + file)



tf.reset_default_graph()
test = TensorFlow_Optimizer(Energy_obj_Funct, Gradient_funct, {'variables':[[0,1,2]], 'constants':[HF_initial_state]},
                            optimizer='GradientDescent', learning_rate=0.55)

test.optimize(20)


x = tf.constant([0,1,2])
x.get_shape().num_elements()


# ## NOTE
# vv = tf.Variable(22, dtype=tf.float32)
# sess = tf.Session()
# init = tf.global_variables_initializer()  # for some reason i need to repeat this!
# sess.run(init)
# print(sess.run(vv))