from scipy.optimize import minimize
import matplotlib.pyplot as plt


class Optimizer:
    """

    Base class for optimizers. To specify a new optimization technique simply define a new objective function

    Args:
        funct (callable): The objective function to be minimized. funt(x, *args) -> float
        X0 (numpy.ndarray): array of size N, where N is is no. of independent variables
        method (str): Type of optimizer
        tol (float, optional): Tolerance for termination.
        store_values (bool, optional): Whether to store obj functions outputs and inputs during optimization
        display_iter_steps (bool, optional): Whether to print each optimization step
        display_convergence_message (bool, optional): Set to True to print convergence messages.
        args (tuple, optional): Extra arguments passed to the objective function [its derivatives]
                                aka: fun, jac and hess functions.



    Attributes:
        iter (int): Number of iterations optimizer performed
        obj_fun_output_values (list): list of function outputs at each optimization step.
        obj_fun_input_vals (list): list of function inputs at each optimization step.

    """

    def __init__(self, funct, X0, method, tol=None, store_values=False, display_iter_steps=False,
                 display_convergence_message=True, args=()):

        self.funct = funct
        self.X0 = X0
        self.args = args
        self.method = method
        self.tol = tol
        self.store_values = store_values
        self.display_iter_steps = display_iter_steps
        self.display_convergence_message = display_convergence_message

        # TODO (can add hess, jac and other gradient functions to this!)



    def callback_store_values(self, xk):
        # Called after each iteration if set to store_values set too True
        if self.args == ():
            val = self.funct(xk)
        else:
            val = self.funct(xk, *self.args)

        if self.store_values is True:
            self.obj_fun_output_values.append(val)
            self.obj_fun_input_vals.append(xk)

        if self.display_iter_steps:
            print(f'{self.iter}: Input_to_Funct: {xk}: Output: {val}')  # self.iters and xk
        self.iter += 1


    #    def AngleBounds(self):
    #        b = (0, 2*math.pi)
    #        bnds = [b for i in range(len(self.initial_guess))]

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
        if self.store_values is True:
            self.iter=0
            self.obj_fun_output_values=[]
            self.obj_fun_input_vals=[]

        elif self.display_iter_steps is True:
            self.iter=0

        options = {'maxiter': max_iter,
                   'disp': self.display_convergence_message}

        kwargs = {'fun': self.funct,
                  'x0': self.X0,  # = param_obj_fun
                  'args': self.args,
                  'method': self.method,
                  'tol': self.tol,
                  'options': options,
                  'callback': self.callback_store_values if self.store_values or self.display_iter_steps is True else None}

        self.optimized_result = minimize(**kwargs)  # scipy.optimize.minimize
        if self.display_convergence_message:
            print(f'Reason for termination is {self.optimized_result.message}')

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
    def quad(x, c):
        return x ** 2 + c
    X0 = 10
    c=2
    GG = Optimizer(quad, X0, 'Nelder-Mead', store_values=True, display_iter_steps=True,  tol=1e-3,  args=[c],
                   display_convergence_message= True)
    GG.get_env(20)
    GG.plot_convergence()



class NEW_Optimizer:
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
                 tol=None, display_convergence_message=False, display_steps=False):

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

        self.display_convergence_message = display_convergence_message
        self.display_steps = display_steps

        # TODO (can add hess, jac and other gradient functions to this!)

    def callback_store_values(self, xk):
        # Called after each iteration if set to store_values set too True

        self.obj_fun_input_vals.append(xk)

        if self.display_steps:
            val = self.funct(xk, *self.args)
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
        self.optimized_result = minimize(**kwargs)  # scipy.optimize.minimize
        if self.display_convergence_message:
            print(self.optimized_result.message)

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
