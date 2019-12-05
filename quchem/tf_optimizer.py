import tensorflow as tf



from quchem.Ansatz_Generator_Functions import *
from quchem.Simulating_Quantum_Circuit import *
from quchem.Unitary_partitioning import *


def Calc_Energy_NORMAL(theta_list, HF_initial_state, num_shots=10000):
    HF_UCC = Full_state_prep_circuit(HF_initial_state, T1_and_T2_theta_list=theta_list)
    HF_UCC.complete_UCC_circuit()
    full_anstaz_circuit = HF_UCC.UCC_full_circuit

    UnitaryPart = UnitaryPartition(anti_commuting_sets, full_anstaz_circuit, S=0)
    UnitaryPart.Get_Quantum_circuits_and_constants()
    quantum_circuit_dict = UnitaryPart.circuits_and_constants

    sim = Simulation_Quantum_Circuit_Dict(quantum_circuit_dict, num_shots)
    Energy = sim.Calc_energy_via_parity()
    Energy = np.float32(Energy.real)
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

def Gradient_funct_NORMAL(theta_guess_list, HF_initial_state):

    theta_list_PLUS_full, theta_list_MINUS_full = Gradient_angles_setup(theta_guess_list)

    partial_gradient_list = []
    for j in range(len(theta_guess_list)):

        # theta = theta_guess_list[j]

        theta_list_PLUS = theta_guess_list
        theta_list_PLUS[j] = theta_list_PLUS_full[j]
        Ham_PLUS = Calc_Energy_NORMAL(theta_list_PLUS, HF_initial_state, num_shots = 10000)



        theta_list_MINUS = theta_guess_list
        theta_list_MINUS[j] = theta_list_MINUS_full[j]
        Ham_MINUS = Calc_Energy_NORMAL(theta_list_MINUS, HF_initial_state, num_shots = 10000)


        Gradient = (Ham_PLUS - Ham_MINUS)  # /2
        partial_gradient_list.append(Gradient)  #append((Gradient, theta))
    return partial_gradient_list


def Calc_Energy_TENSOR(theta_list_TENSOR, HF_initial_state, num_shot):

    sess = tf.Session()
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    theta_list = [sess.run(theta) for theta in theta_list_TENSOR]

    output = Calc_Energy_NORMAL(theta_list, HF_initial_state, num_shots=num_shot)
    return tf.Variable(output, dtype=tf.float64)

def Gradient_funct_TENSOR(theta_list_TENSOR, HF_initial_state):
    sess = tf.Session()
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    theta_list = [sess.run(theta) for theta in theta_list_TENSOR]

    gradient_list = Gradient_funct_NORMAL(theta_list, HF_initial_state)
    return zip(gradient_list, theta_list_TENSOR)


tf.reset_default_graph()
max_iter = 50
num_shots = 1000

theta_guess = [random.uniform(0, 2*math.pi) for i in range(3)]
#theta_guess = [1.55957373, 1.57789987, 0.78561344]

theta_list_TENSOR = [tf.Variable(i, dtype=tf.float32) for i in theta_guess]


function = Calc_Energy_TENSOR(theta_list_TENSOR, HF_initial_state, num_shots)
grads_and_vars = Gradient_funct_TENSOR(theta_list_TENSOR, HF_initial_state)

#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)

train = optimizer.apply_gradients(grads_and_vars)

init = tf.compat.v1.global_variables_initializer()
E_list = []
Angle_list = []

with tf.compat.v1.Session() as session:
    session.run(init)
    Angles = [session.run(var) for var in theta_list_TENSOR]
    Energy = session.run(function)
    print("starting at variables:", Angles, "Energy:", Energy)
    E_list.append(Energy)
    Angle_list.append(Angles)

    for step in range(max_iter):

        grads_and_vars = Gradient_funct_TENSOR(theta_list_TENSOR, HF_initial_state)
       # train = optimizer.apply_gradients(grads_and_vars)

        yy = list(grads_and_vars)
        ww = [(gradient, session.run(theta_Tensor)) for gradient, theta_Tensor in yy]
        print(ww)
        train = optimizer.apply_gradients(yy)

        # init = tf.compat.v1.global_variables_initializer()    # <- seems WRONG
        # session.run(init)                                     # <- seems WRONG
        session.run(train)

        Angles = [session.run(angle) for angle in theta_list_TENSOR]
        Energy = Calc_Energy_NORMAL(Angles, HF_initial_state, num_shots=num_shots)  # NOTE this is not the tensor function
        # Energy = session.run(function) <--- note it is NOT THIS!
        print("step", step, "variables:", Angles, "Energy:", Energy)
        E_list.append(Energy)
        Angle_list.append(Angles)

# def Function_to_minimise(x, y, const):
#     # z = x^2 + y^2 + constant
#     z = x ** 2 + y ** 2 + const
#     return z
#
#
# def calc_grad(x, y):
#     # z = 2x^2 + y^2 + constant
#     dz_dx = 2 * x
#     dz_dy = 2 * y
#
#     return dz_dx, dz_dy
#
#
# def Function_to_minimise_TENSOR(a, b, const):
#     sess = tf.Session()
#     init = tf.compat.v1.global_variables_initializer()
#     sess.run(init)
#
#     x = sess.run(a)
#     y = sess.run(b)
#
#     output = Function_to_minimise(x, y, const)
#     return tf.Variable(output, dtype=tf.float64)
#
#
# def calcu_grad_TENSOR(a, b):
#     sess = tf.Session()
#     init = tf.compat.v1.global_variables_initializer()
#     sess.run(init)
#
#     x = sess.run(a)
#     y = sess.run(b)
#     dz_dx, dz_dy = calc_grad(a, b)
#     return [(dz_dx, a), (dz_dy, b)]