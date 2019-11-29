import tensorflow as tf


class TensorFlow_Optimizer():

    def __init__(self,function_to_minimise, Gradient_function, kwargs):
        self.function_to_minimise = function_to_minimise
        self.kwargs = kwargs
        self.Gradient_function = Gradient_function

    def Get_functions_args_as_tensors(self):
        for key in self.kwargs:
            if key == 'constants':
                constants = [tf.constant(const, dtype=tf.float32) for const in self.kwargs[key]]
            elif key == 'variables':
                vars = [tf.Variable(const, dtype=tf.float32) for const in self.kwargs[key]]

        return self.function_to_minimise(*vars, *constants), vars

    def Calc_Gradient(self, vars):
        return self.Gradient_function(*vars)


    def optimize(self):

        function, variables = self.Get_functions_args_as_tensors()
        grads_and_vars = self.Calc_Gradient(variables)

        optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.55)
        train = optimizer.apply_gradients(grads_and_vars)

        init = tf.compat.v1.global_variables_initializer()

        with tf.compat.v1.Session() as session:
            session.run(init)
            print("starting at angles:", [session.run(var)for var in variables],  "Energy:", session.run(function))
            for step in range(10):
                session.run(train)
                print("step", step, "Angles:", [session.run(var)for var in variables], "Energy:", session.run(function))



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

    test = TensorFlow_Optimizer(function_to_minimize, calc_grad, input_dict)
    test.optimize()