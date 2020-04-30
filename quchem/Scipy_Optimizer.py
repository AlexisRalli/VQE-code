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


