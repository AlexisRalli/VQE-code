from scipy.optimize import minimize
import cirq
import matplotlib.pyplot as plt

if __name__ == '__main__':
    from Simulating_Quantum_Circuit import *
else:
    from .Simulating_Quantum_Circuit import *

class Optimizer:
    '''
    Base class for optimizers. To specify a new optimization technique simply define a new objective function
    '''

    def __init__(self, num_shots, theta_guess_list,
                noisy=True, store_values=False, optimized_result=None):

        self.num_shots = num_shots
        self.initial_guess = theta_guess_list


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
        from NEW_SIM_Class import simulation_VQE, simulation_Result

        param_resolver = self.parameter_setup(param_obj_fun, self.circuit_label_dictionary)

        VQE_run = simulation_VQE(self.Full_instance_Circuit_List, param_resolver, self.num_shots,
                                 self.Hamiltonian_ConstantList, self.IdentityConstant)
        Raw_Results_list = VQE_run.Simulate()

        VQE_run_results = simulation_Result(self.Hamiltonian_QubitNoList, self.Hamiltonian_OperationList,
                                            self.Hamiltonian_ConstantList,
                                            self.IdentityConstant,
                                            Raw_Results_list, VQE_run.num_shots)

        Energy = VQE_run_results.Total_Energy()

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