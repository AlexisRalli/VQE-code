from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd

from quchem.Ansatz_Generator_Functions import *
from quchem.Simulating_Quantum_Circuit import *
from quchem.Unitary_partitioning import *


class Optimizer:
    '''
    Base class for optimizers. To specify a new optimization technique simply define a new objective function
    '''

    def __init__(self, num_shots, theta_guess_list, HF_initial_state, anti_commuting_sets,
                noisy=True, store_values=False, optimized_result=None):

        self.num_shots = num_shots
        self.initial_guess = theta_guess_list
        self.HF_initial_state = HF_initial_state


        self.iters = 0
        self.reps = num_shots
        self.obj_fun_values = []
        self.noisy = noisy
        self.store_values = store_values
        self.optimized_result = optimized_result
        self.theta_params = None
        self.anti_commuting_sets = anti_commuting_sets

        # self.log_ANGLES = True
        # self.store_values_ANGLES = [] #can append to here

    def set_noise(self, _bool):
        self.noisy = _bool
        self.store_values = _bool


    def set_reps(self, reps):
        self.reps = reps


    def objective_function(self, param_obj_fun):
        # Is full Objective function, with all parameters
        """"
        Returns Energy value... to be minimized!

         """

        HF_UCC = Full_state_prep_circuit(self.HF_initial_state, T1_and_T2_theta_list=param_obj_fun)
        HF_UCC.complete_UCC_circuit()
        full_anstaz_circuit = HF_UCC.UCC_full_circuit


        UnitaryPart = UnitaryPartition(self.anti_commuting_sets, full_anstaz_circuit, S=0)
        UnitaryPart.Get_Quantum_circuits_and_constants()
        quantum_circuit_dict = UnitaryPart.circuits_and_constants

        sim = Simulation_Quantum_Circuit_Dict(quantum_circuit_dict, self.num_shots)
        Energy = sim.Calc_energy_via_parity()

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
