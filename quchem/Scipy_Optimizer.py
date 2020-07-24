from scipy.optimize import minimize
import matplotlib.pyplot as plt


# class Optimizer:
#     """
#
#     Base class for optimizers. To specify a new optimization technique simply define a new objective function
#
#     Args:
#         funct (callable): The objective function to be minimized. funt(x, *args) -> float
#         X0 (numpy.ndarray): array of size N, where N is is no. of independent variables
#         method (str): Type of optimizer
#         tol (float, optional): Tolerance for termination.
#         store_values (bool, optional): Whether to store obj functions outputs and inputs during optimization
#         display_iter_steps (bool, optional): Whether to print each optimization step
#         display_convergence_message (bool, optional): Set to True to print convergence messages.
#         args (tuple, optional): Extra arguments passed to the objective function [its derivatives]
#                                 aka: fun, jac and hess functions.
#
#
#
#     Attributes:
#         iter (int): Number of iterations optimizer performed
#         obj_fun_output_values (list): list of function outputs at each optimization step.
#         obj_fun_input_vals (list): list of function inputs at each optimization step.
#
#     """
#
#     def __init__(self, funct, X0, method, tol=None, store_values=False, display_iter_steps=False,
#                  display_convergence_message=True, args=()):
#
#         self.funct = funct
#         self.X0 = X0
#         self.args = args
#         self.method = method
#         self.tol = tol
#         self.store_values = store_values
#         self.display_iter_steps = display_iter_steps
#         self.display_convergence_message = display_convergence_message
#
#         # TODO (can add hess, jac and other gradient functions to this!)
#
#
#
#     def callback_store_values(self, xk):
#         # Called after each iteration if set to store_values set too True
#         if self.args == ():
#             val = self.funct(xk)
#         else:
#             val = self.funct(xk, *self.args)
#
#         if self.store_values is True:
#             self.obj_fun_output_values.append(val)
#             self.obj_fun_input_vals.append(xk)
#
#         if self.display_iter_steps:
#             print(f'{self.iter}: Input_to_Funct: {xk}: Output: {val}')  # self.iters and xk
#         self.iter += 1
#
#
#     #    def AngleBounds(self):
#     #        b = (0, 2*math.pi)
#     #        bnds = [b for i in range(len(self.initial_guess))]
#
#     def get_env(self, max_iter):
#         """
#
#        Function that preforms optimization step
#
#         Args:
#             max_iter (int): Maximum number of iterations to perform. Depending on the method each iteration may
#                             use several function evaluations.
#
#         Attributes:
#             iter (int): Number of iterations optimizer performed
#             obj_fun_output_values (list): list of function outputs at each optimization step.
#             obj_fun_input_vals (list): list of function inputs at each optimization step.
#
#         Returns:
#             self.optimized_result (scipy.optimize.optimize.OptimizeResult): OptimizeResult object.
#
#         """
#         if self.store_values is True:
#             self.iter=0
#             self.obj_fun_output_values=[]
#             self.obj_fun_input_vals=[]
#
#         elif self.display_iter_steps is True:
#             self.iter=0
#
#         options = {'maxiter': max_iter,
#                    'disp': self.display_convergence_message}
#
#         kwargs = {'fun': self.funct,
#                   'x0': self.X0,  # = param_obj_fun
#                   'args': self.args,
#                   'method': self.method,
#                   'tol': self.tol,
#                   'options': options,
#                   'callback': self.callback_store_values if self.store_values or self.display_iter_steps is True else None}
#
#         self.optimized_result = minimize(**kwargs)  # scipy.optimize.minimize
#         if self.display_convergence_message:
#             print(f'Reason for termination is {self.optimized_result.message}')
#
#     def plot_convergence(self):  # , file):
#
#
#         if self.store_values is False:
#             raise NotImplementedError('Cannot plot convergence as values at each opt. step NOT stored')
#         else:
#             # dir_path = os.path.dirname(os.path.realpath(__file__))
#             plt.figure()
#             x = list(range(len(self.obj_fun_output_values)))
#             plt.plot(x, self.obj_fun_output_values)
#             plt.xlabel('iterations')
#             plt.ylabel('objective function value')
#             # plt.savefig(dir_path + '/' + file)

# if __name__ == '__main__':
#     def quad(x, c):
#         return x ** 2 + c
#     X0 = 10
#     c=2
#     GG = Optimizer(quad, X0, 'Nelder-Mead', store_values=True, display_iter_steps=True,  tol=1e-3,  args=[c],
#                    display_convergence_message= True)
#     GG.get_env(20)
#     GG.plot_convergence()



class Optimizer:
    """

    Base class for optimizers. To specify a new optimization technique simply define a new objective function

    Args:
        func (callable): The objective function to be minimized. funt(x, *args) -> float
        X0 (numpy.ndarray): initial guess
        args (tuple, optional): optional input arguements to func (NOTE these will not be optimized!) and jac, hess etc
        method (str): Type of optimizer... custom method can be defined too
        jac (callable): Calculates first derivative of objective function to be minimized. jac(x, *args) -> N array
                        note keywords '2-point', '3-point', 'cs' can be used for FINITE DIFFERENCE

        hess (callable, optional): Calculates Hessian matrix of objective function to be minimized. hess(x, *args) -> N x N array
                        note keywords '2-point', '3-point', 'cs' can be used for FINITE DIFFERENCE

        hessp (callable, optional): Hessian of function TIMES vector p. Returns Hess(x,p, *args) -> N array
                                    where p is of dimention N.

        bounds (sequence, optional): sequence of tuples [(lower, upper), (lower, upper) ... ] for input varibles

        constraints(dict, optional): either stand along dict or list of dictionaries (for multiple constraints)
                                     See online for further info

        tol (float, optional): Tolerance for termination.

        display_iter_steps (bool, optional): Whether to print each optimization step
                                             NOTE THIS RUNS objective function... therefore can have a cost!

        display_convergence_message (bool, optional): Set to True to print convergence messages.


    Attributes:
        iter (int): Number of iterations optimizer performed
        obj_fun_output_values (list): list of function outputs at each optimization step.
        obj_fun_input_vals (list): list of function inputs at each optimization step.

    """

    def __init__(self, func, X0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=None,
                 tol=None, display_convergence_message=False, display_steps=False, custom_optimizer_DICT=None):

        self.func = func
        self.X0 = X0
        self.args = args
        self.method = method
        self.jac = jac
        self.hess= hess
        self.hessp = hessp
        self.bounds=bounds
        self.constraints = constraints
        self.tol = tol
        self.custom_optimizer_DICT = custom_optimizer_DICT

        self.display_convergence_message = display_convergence_message
        self.display_steps = display_steps

        # TODO (can add hess, jac and other gradient functions to this!)

    def callback_store_values(self, xk):
        # Called after each iteration if set to store_values set too True

        self.obj_fun_input_vals.append(xk)

        if self.display_steps:
            val = self.func(xk, *self.args)
            self.obj_fun_output_values.append(val)

            print(f'{self.iter}: Input_to_Funct: {xk}: Output: {val}')  # self.iters and xk
            self.iter += 1

    def get_env(self, max_iter):
        """

       Function that preforms optimization step

        Args:
            max_iter (int): Maximum number of iterations to perform. Depending on the method each iteration may
                            use several function evaluations.

        Attributes:
            iter (int): Number of iterations optimizer performed
            obj_fun_output_values (list): list of function outputs at each optimization step.
            obj_fun_input_vals (list): list of function inputs at each optimization step.

        Returns:
            self.optimized_result (scipy.optimize.optimize.OptimizeResult): OptimizeResult object.

        """

        if self.display_steps is True:
            self.iter=0
            self.obj_fun_output_values = []

        options = {'maxiter': max_iter,
                   'disp': self.display_convergence_message}

        kwargs = {'fun': self.func,
                  'x0': self.X0,  # = param_obj_fun
                  'args': self.args,
                  'method': self.method,
                  'jac': self.jac,
                  'hess': self.hess,
                  'hessp': self.hessp,
                  'bounds': self.bounds,
                  'constraints': self.constraints,
                  'tol': self.tol,
                  'options': options,
                  'callback': self.callback_store_values }#if self.display_steps is True else None}


        self.obj_fun_input_vals = []

        if self.custom_optimizer_DICT:
            kwargs['options'] = {**self.custom_optimizer_DICT, **options}

        self.optimized_result = minimize(**kwargs)  # scipy.optimize.minimize


        if self.display_convergence_message:
            print(self.optimized_result.message)

    def plot_convergence(self):  # , file):

        if self.display_steps is True:
            # dir_path = os.path.dirname(os.path.realpath(__file__))
            plt.figure()
            x = list(range(len(self.obj_fun_output_values)))
            plt.plot(x, self.obj_fun_output_values)
            plt.xlabel('iterations')
            plt.ylabel('objective function value')
            # plt.savefig(dir_path + '/' + file)
        else:
            raise NotImplementedError('Cannot plot convergence as values at each opt. step NOT stored')



## CUSTOM ADAM OPTIMIZER
import numpy as np
from scipy.optimize.optimize import OptimizeResult, wrap_function, _status_message, _check_unknown_options, _approx_fprime_helper
from numpy import asarray

def _minimize_Adam(func, x0, args=(), jac=None, bounds=None, constraints=None,
                 tol=None, learning_rate=0.001, beta_1=0.9, beta_2=0.999, delta=1e-8, maxiter=500, disp=False,
                   maxfev=15000, callback=None, epsilon=1e-8,  **unkown_options):
    """

    Minimize function using the Adam Algorithm.

    https://arxiv.org/abs/1412.6980


    Args:
        func (callable): The objective function to be minimized. funt(x, *args) -> float
        X0 (numpy.ndarray): initial guess
        args (tuple, optional): optional input arguements to func (NOTE these will not be optimized!) and jac, hess etc
        method (str): Type of optimizer... custom method can be defined too
        jac (callable): Calculates first derivative of objective function to be minimized. jac(x, *args) -> N array
                        if set to None then uses FINITE DIFFERENCE!

        delta (float): finite diffence gradient step

        tol (float, optional): Tolerance for termination.

        learning_rate (float): Step size
        beta_1 (float): The exponential decay rate for the 1st moment estimates.
        beta_2 (float):  The exponential decay rate for the 2nd moment estimates.
        epsilon (float):  Constant (small) for numerical stability

    Attributes:
        t (int): Timestep
        m_t (float): first moment vector
        v_t (float): second moment vector

    # TODO add bounds and constraints!
    # TODO consider making xtol and ftol variables (aka function and variable convergence check... rather than one global
    # check

    """

    x0 = asarray(x0).ravel()

    num_FUNCT_eval, FUNCT = wrap_function(func, args)

    if jac is None:
        num_JAC_eval=0
        def funct_and_grad(x):
            f = FUNCT(x, *args)
            g = _approx_fprime_helper(x, FUNCT, delta)
            return f, g
    else:
        num_JAC_eval, FPRIME = wrap_function(jac, args)
        def funct_and_grad(x):
            f = FUNCT(x, *args)
            g = FPRIME(x, *args)
            return f,g

    # initialization
    t = 0  # timestep
    m_t = 0  # 1st moment vector
    v_t = 0  # 2nd moment vector
    X_t = x0

    n_iterations=0
    while True:
        n_iterations+=1

        # ADAM Algorithm
        t += 1
        f_t, g_t = funct_and_grad(X_t)
        m_t = beta_1 * m_t + (
                    1 - beta_1) * g_t  # updates the moving averages of the gradient (biased first moment estimate)
        v_t = beta_2 * v_t + (1 - beta_2) * (
                    g_t * g_t)  # updates the moving averages of the squared gradient (biased 2nd
        # raw moment estimate)

        m_cap = m_t / (1 - (beta_1 ** t))  # Compute bias-corrected first moment estimate
        v_cap = v_t / (1 - (beta_2 ** t))  # Compute bias-corrected second raw moment estimate
        X_t_prev = X_t

        X_t = X_t_prev - (learning_rate * m_cap) / (np.sqrt(v_cap) + epsilon)  # updates the parameters
        # Adam END

        if callback is not None:
            callback(np.copy(X_t))


        # check for termination
        if n_iterations>1:
            if np.isclose(f_t_prev, f_t, rtol=tol).all():  # checks if FUNCTION has converged
                break
        f_t_prev = f_t


        if np.isclose(X_t, X_t_prev, rtol=tol).all():  # checks if VARIABLES have converged
            break

        if num_FUNCT_eval[0] >= maxfev: # checks if overdone too many function evaluations
            break

        if n_iterations >= maxiter: # checks number of iterations
            break


    warnflag = 0
    if num_FUNCT_eval[0] >= maxfev:
        warnflag = 1
        msg = _status_message['maxfev']
        if disp:
            print("Warning: " + msg)
    elif n_iterations >= maxiter:
        warnflag = 2
        msg = _status_message['maxiter']
        if disp:
            print("Warning: " + msg)

    else:
        if disp:
            msg = _status_message['success']
            print(msg)
            print("         Current function value: {}".format(f_t))
            print("         Iterations: %d" % n_iterations)
            print("         Function evaluations: %d" % num_FUNCT_eval[0])
            print("         Function evaluations:{}".format(num_JAC_eval))

    result = OptimizeResult(fun=f_t, nit=n_iterations, nfev=num_FUNCT_eval, njev=num_JAC_eval,
                            status=warnflag, success=(warnflag == 0),
                            message=msg, x=X_t)
    return result


if __name__ == '__main__':
    def Function_to_minimise(input_vect, *args):
        # z = x^2 + y^2 + constant
        x = input_vect[0]
        y = input_vect[1]
        z = x ** 2 + y ** 2 + args
        return z

    def calc_grad(input_vect, *args):
        # z = 2x^2 + y^2 + constant
        x = input_vect[0]
        y = input_vect[1]

        dz_dx = 2 * x
        dz_dy = 2 * y
        return np.array([dz_dx, dz_dy])

    X0 = np.array([1,2])
    arg = (2,)

    custom_optimizer_DICT = {'learning_rate': 0.01, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8,
                               'delta': 1e-8, 'maxfev': 15000}

    x = Optimizer(Function_to_minimise, X0, args=arg, method=_minimize_Adam, jac=calc_grad, hess=None, hessp=None,
                      bounds=None, constraints=None,tol=1e-20, display_convergence_message=True, display_steps=True, custom_optimizer_DICT=custom_optimizer_DICT)
    x.get_env(5000)

    print(x.optimized_result)

# def Adam_Opt(X_0, function, gradient_function, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, max_iter=500,
#              disp=False, tolerance=1e-5, store_steps=False):
#     """
#
#     To be passed into Scipy Minimize method
#
#     https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
#
#
#     https://github.com/sagarvegad/Adam-optimizer/blob/master/Adam.py
#     https://arxiv.org/abs/1412.6980
#     Args:
#         function (callable): Stochastic objective function
#         gradient_function (callable): function to obtain gradient of Stochastic objective
#         X0 (np.array):  Initial guess
#         learning_rate (float): Step size
#         beta_1 (float): The exponential decay rate for the 1st moment estimates.
#         beta_2 (float):  The exponential decay rate for the 2nd moment estimates.
#         epsilon (float):  Constant (small) for numerical stability
#
#     Attributes:
#         t (int): Timestep
#         m_t (float): first moment vector
#         v_t (float): second moment vector
#
#     """
#     input_vectors=[]
#     output_results=[]
#
#     # initialization
#     t=0  # timestep
#     m_t = 0 #1st moment vector
#     v_t = 0 #2nd moment vector
#     X_t = X_0
#
#     while(t<max_iter):
#
#         if store_steps is True:
#             input_vectors.append(X_t)
#             output_results.append(function(X_t))
#
#         t+=1
#         g_t = gradient_function(X_t)
#         m_t = beta_1*m_t + (1-beta_1)*g_t	#updates the moving averages of the gradient (biased first moment estimate)
#         v_t = beta_2*v_t + (1-beta_2)*(g_t*g_t)	#updates the moving averages of the squared gradient (biased 2nd
#                                                 # raw moment estimate)
#
#         m_cap = m_t / (1 - (beta_1 ** t))  # Compute bias-corrected first moment estimate
#         v_cap = v_t / (1 - (beta_2 ** t))  # Compute bias-corrected second raw moment estimate
#         X_t_prev = X_t
#         X_t = X_t_prev - (learning_rate * m_cap) / (np.sqrt(v_cap) + epsilon)  # updates the parameters
#
#         if disp is True:
#             output = function(X_t)
#             print('step: {} input:{} obj_funct: {}'.format(t, X_t, output))
#
#         if np.isclose(X_t, X_t_prev, atol=tolerance).all(): # convergence check
#             break
#     if store_steps is True:
#         return X_t, input_vectors, output_results
#     else:
#         return X_t
#
# if __name__ == '__main__':
#     def Function_to_minimise(input_vect, const=2):
#         # z = x^2 + y^2 + constant
#         x = input_vect[0]
#         y = input_vect[1]
#         z = x ** 2 + y ** 2 + const
#         return z
#
#     def calc_grad(input_vect):
#         # z = 2x^2 + y^2 + constant
#         x = input_vect[0]
#         y = input_vect[1]
#
#         dz_dx = 2 * x
#         dz_dy = 2 * y
#         return np.array([dz_dx, dz_dy])
#
#     X0 = np.array([1,2])
#     GG = Adam_Opt(X0, calc_grad,
#                   learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
#
#
#     print(Function_to_minimise(GG))
#
#     import matplotlib.pyplot as plt
#     from matplotlib import cm
#     from mpl_toolkits.mplot3d import Axes3D
#     import numpy as np
#
#     x = np.arange(-10, 10, 0.25)
#     y = np.arange(-10, 10, 0.25)
#     const = 2
#
#     x, y = np.meshgrid(x, y)
#     z = x ** 2 + y ** 2 + const
#
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.viridis)
#     plt.show()
#     print('Minimum should be:', 2.0)


class Optimizer_save_no_measurements:
    """

    Base class for optimizers. To specify a new optimization technique simply define a new objective function

    Args:
        func (callable): The objective function to be minimized. funt(x, *args) -> float
        X0 (numpy.ndarray): initial guess
        args (tuple, optional): optional input arguements to func (NOTE these will not be optimized!) and jac, hess etc
        method (str): Type of optimizer... custom method can be defined too
        jac (callable): Calculates first derivative of objective function to be minimized. jac(x, *args) -> N array
                        note keywords '2-point', '3-point', 'cs' can be used for FINITE DIFFERENCE

        hess (callable, optional): Calculates Hessian matrix of objective function to be minimized. hess(x, *args) -> N x N array
                        note keywords '2-point', '3-point', 'cs' can be used for FINITE DIFFERENCE

        hessp (callable, optional): Hessian of function TIMES vector p. Returns Hess(x,p, *args) -> N array
                                    where p is of dimention N.

        bounds (sequence, optional): sequence of tuples [(lower, upper), (lower, upper) ... ] for input varibles

        constraints(dict, optional): either stand along dict or list of dictionaries (for multiple constraints)
                                     See online for further info

        tol (float, optional): Tolerance for termination.

        display_iter_steps (bool, optional): Whether to print each optimization step
                                             NOTE THIS RUNS objective function... therefore can have a cost!

        display_convergence_message (bool, optional): Set to True to print convergence messages.


    Attributes:
        iter (int): Number of iterations optimizer performed
        obj_fun_output_values (list): list of function outputs at each optimization step.
        obj_fun_input_vals (list): list of function inputs at each optimization step.

    """

    def __init__(self, func, X0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=None,
                 tol=None, display_convergence_message=False, display_steps=False, custom_optimizer_DICT=None):

        self.func = func
        self.X0 = X0
        self.args = args
        self.method = method
        self.jac = jac
        self.hess= hess
        self.hessp = hessp
        self.bounds=bounds
        self.constraints = constraints
        self.tol = tol
        self.custom_optimizer_DICT = custom_optimizer_DICT
        self.total_measurements = []

        self.display_convergence_message = display_convergence_message
        self.display_steps = display_steps

        # TODO (can add hess, jac and other gradient functions to this!)

    def callback_store_values(self, xk):
        # Called after each iteration if set to store_values set too True

        self.obj_fun_input_vals.append(xk)

        if self.display_steps:
            val = self.func(xk, *self.args)
            self.obj_fun_output_values.append(val)

            print(f'{self.iter}: Input_to_Funct: {xk}: Output: {val}')  # self.iters and xk
            self.iter += 1

    def get_env(self, max_iter):
        """

       Function that preforms optimization step

        Args:
            max_iter (int): Maximum number of iterations to perform. Depending on the method each iteration may
                            use several function evaluations.

        Attributes:
            iter (int): Number of iterations optimizer performed
            obj_fun_output_values (list): list of function outputs at each optimization step.
            obj_fun_input_vals (list): list of function inputs at each optimization step.

        Returns:
            self.optimized_result (scipy.optimize.optimize.OptimizeResult): OptimizeResult object.

        """

        if self.display_steps is True:
            self.iter=0
            self.obj_fun_output_values = []

        options = {'maxiter': max_iter,
                   'disp': self.display_convergence_message}

        kwargs = {'fun': self.func,
                  'x0': self.X0,  # = param_obj_fun
                  'args': self.args,
                  'method': self.method,
                  'jac': self.jac,
                  'hess': self.hess,
                  'hessp': self.hessp,
                  'bounds': self.bounds,
                  'constraints': self.constraints,
                  'tol': self.tol,
                  'options': options,
                  'callback': self.callback_store_values }#if self.display_steps is True else None}


        self.obj_fun_input_vals = []

        if self.custom_optimizer_DICT:
            kwargs['options'] = {**self.custom_optimizer_DICT, **options}

        self.optimized_result = minimize(**kwargs)  # scipy.optimize.minimize


        if self.display_convergence_message:
            print(self.optimized_result.message)

    def plot_convergence(self):  # , file):

        if self.display_steps is True:
            # dir_path = os.path.dirname(os.path.realpath(__file__))
            plt.figure()
            x = list(range(len(self.obj_fun_output_values)))
            plt.plot(x, self.obj_fun_output_values)
            plt.xlabel('iterations')
            plt.ylabel('objective function value')
            # plt.savefig(dir_path + '/' + file)
        else:
            raise NotImplementedError('Cannot plot convergence as values at each opt. step NOT stored')