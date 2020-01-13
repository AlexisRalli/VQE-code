import matplotlib.pyplot as plt
import tensorflow as tf


class Tensor_Flow_Optimizer:
    """

    Base class for TensorFlow optimizers. To specify a new optimization technique simply define a new objective function

    Args:
        funct (callable): The objective function to be minimized. funt(x, *args) -> float
        X0 (numpy.ndarray): array of size N, where N is is no. of independent variables
        grad_funct (callable): Function to calculate gradient. Note MUST RETURN list of tuples of partial deriv and term
                               grad_funct(x, *args) -> [(dz_dx, x), (dz_dy, y), ... etc...]
        method (str): Type of optimizer
        learning_rate (float, optional):
        beta1 (float, optional):
        beta2 (float, optional):
        args (tuple, optional): Extra arguments passed to the objective function [its derivatives]
                        aka: fun, jac and hess functions.
        store_values (bool, optional): Whether to store obj functions outputs and inputs during optimization
        display_iter_steps (bool, optional): Whether to print each optimization step

    Attributes:
        obj_fun_output_values (list): list of function outputs at each optimization step.
        obj_fun_input_vals (list): list of function inputs at each optimization step.

    """

    def __init__(self, funct, X0, method, grad_funct, learning_rate=0.01, beta1=0.9,
                 beta2=0.999, args=(), store_values=True, display_iter_steps=True):

        self.funct = funct
        self.X0 = X0
        self.args = args
        self.method = method
        self.grad_funct = grad_funct
        self.beta1 = beta1
        self.beta2 = beta2
        self.learning_rate = learning_rate
        self.store_values =store_values
        self.display_iter_steps = display_iter_steps


        # tf.reset_default_graph()
        tf.compat.v1.reset_default_graph()

    def object_function_TENSOR(self, Input_TENSOR_vector):
        """

       Takes in list of tensor inputs, initialises these values and returns the output of objective function
       as a tensorflow variable tensor.

        Args:
            Input_TENSOR_vector (list): Takes in list of tensor values.

        Returns:
            output of objective function to be minimized, as a TensorFlow variable.

        """
        sess = tf.Session()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)
        Input_vector = [sess.run(i) for i in Input_TENSOR_vector]

        if self.args==():
            output = self.funct(Input_vector)
        else:
            output = self.funct(Input_vector, *self.args)
        return tf.Variable(output, dtype=tf.float64)

    def Get_Gradient_PLACEHOLDERS(self):
        """

       Generates a list of tensorflow placeholders from initial guess vector.

        Returns:
            list of TensorFlow placeholders.

        """
        place_holder_list = [tf.placeholder(tf.float32, shape=(), name="{}".format(i)) for i in self.X0]
        return place_holder_list

    def Feed_dictionary(self, placeholder_list, grads_and_vars):
        # grads_and_vars is list of tuples of (Gradient, INPUT)
        """

       Function generates feed dictionary for training step: train_step.run(feed_dict=OUTPUT_OF_THIS_FUNCTION).
       Uses placeholders from Get_Gradient_PLACEHOLDERS function and generates dict of {placeholder: gradient}

        Args:
            placeholder_list (list): List of tensorflow placeholders
            grads_and_vars (list): List of tuples, which are (partial_deriv, value): [(dz_dx, x), (dz_dy, y), ...etc.]

        Returns:
            feed_dict (dict): Feed dictionary for tensorflow train

        """
        feed_dict = {}
        for i in range(len(placeholder_list)):
            Place_Holder = placeholder_list[i]
            gradient = grads_and_vars[i][0] # only want gradient!
            feed_dict.update({Place_Holder: gradient})
        return feed_dict

    def optimize(self, max_iter):
        """

       Function that performs optimization

        Args:
            max_iter (int): Max number of iterations for optimizer

        """
        # tf.reset_default_graph()
        tf.compat.v1.reset_default_graph()

        if self.store_values is True:
            self.obj_fun_output_values=[]
            self.obj_fun_input_vals=[]

        # tf.compat.v1.train.GradientDescentOptimizer
        if self.method == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1, beta2=self.beta2)
        elif self.method == 'GradientDescent':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        else:
            raise ValueError('Optimizer Not defined')

        Input_TENSOR_vector = [tf.Variable(i, dtype=tf.float32) for i in self.X0]
        Obj_funct_output = self.object_function_TENSOR(Input_TENSOR_vector)
        _, variables = zip(*optimizer.compute_gradients(Obj_funct_output))

        place_holder_list = self.Get_Gradient_PLACEHOLDERS()

        train_step = optimizer.apply_gradients(zip(place_holder_list, variables))


        with tf.Session() as session:

            if self.args == ():
                output = self.funct(self.X0)
            else:
                output = self.funct(self.X0, *self.args)

            if self.display_iter_steps is True:
                print("starting input:", self.X0, "obj funct out:", output)

            if self.store_values is True:
                self.obj_fun_output_values.append(output)
                self.obj_fun_input_vals.append(self.X0)

            init = tf.compat.v1.global_variables_initializer()
            session.run(init)
            for step in range(max_iter):
                input_vect = [session.run(theta_tensor) for theta_tensor in Input_TENSOR_vector]
                grads_and_vars = self.grad_funct(input_vect)
                grad_sub_dict = self.Feed_dictionary(place_holder_list, grads_and_vars)
                train_step.run(feed_dict=grad_sub_dict)
                NEW_input_vect = [session.run(input) for input in Input_TENSOR_vector]
                if self.args == ():
                    obj_fun_output = self.funct(NEW_input_vect)
                else:
                    obj_fun_output = self.funct(NEW_input_vect, *self.args)

                if self.display_iter_steps is True:
                    print("step", step, "INPUT:", NEW_input_vect, "OUTPUT:", obj_fun_output)

                if self.store_values is True:
                    self.obj_fun_output_values.append(obj_fun_output)
                    self.obj_fun_input_vals.append(NEW_input_vect)

    def plot_convergence(self):  # , file):


        if self.store_values is False:
            raise NotImplementedError('Cannot plot convergence as values at each opt. step NOT stored')
        else:
            # dir_path = os.path.dirname(os.path.realpath(__file__))
            plt.figure()
            x = list(range(len(self.obj_fun_output_values)))
            plt.plot(x, self.obj_fun_output_values)
            plt.xlabel('iterations')
            plt.ylabel('objective function value')
            # plt.savefig(dir_path + '/' + file)

if __name__ == '__main__':
    def Function_to_minimise(input_vect, const):
        # z = x^2 + y^2 + constant
        x = input_vect[0]
        y = input_vect[1]
        z = x ** 2 + y ** 2 + const
        return z

    def calc_grad(input_vect):
        # z = 2x^2 + y^2 + constant
        x = input_vect[0]
        y = input_vect[1]

        dz_dx = 2 * x
        dz_dy = 2 * y
        return [(dz_dx, x), (dz_dy, y)]
    X0 = [1,2]
    const = [2]
    GG = Tensor_Flow_Optimizer(Function_to_minimise, X0, 'Adam', calc_grad, learning_rate=0.1, beta1=0.9,
                                beta2=0.999, args=const, store_values=True, display_iter_steps=True)
    GG.optimize(50)
    GG.plot_convergence()