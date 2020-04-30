import numpy as np

def Adam_Opt(X_0, function, gradient_function, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, max_iter=500,
             disp=False, tolerance=1e-5, store_steps=False):
    """

    To be passed into Scipy Minimize method

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize


    https://github.com/sagarvegad/Adam-optimizer/blob/master/Adam.py
    https://arxiv.org/abs/1412.6980
    Args:
        function (callable): Stochastic objective function
        gradient_function (callable): function to obtain gradient of Stochastic objective
        X0 (np.array):  Initial guess
        learning_rate (float): Step size
        beta_1 (float): The exponential decay rate for the 1st moment estimates.
        beta_2 (float):  The exponential decay rate for the 2nd moment estimates.
        epsilon (float):  Constant (small) for numerical stability

    Attributes:
        t (int): Timestep
        m_t (float): first moment vector
        v_t (float): second moment vector

    """
    input_vectors=[]
    output_results=[]

    # initialization
    t=0  # timestep
    m_t = 0 #1st moment vector
    v_t = 0 #2nd moment vector
    X_t = X_0

    while(t<max_iter):

        if store_steps is True:
            input_vectors.append(X_t)
            output_results.append(function(X_t))

        t+=1
        g_t = gradient_function(X_t)
        m_t = beta_1*m_t + (1-beta_1)*g_t	#updates the moving averages of the gradient (biased first moment estimate)
        v_t = beta_2*v_t + (1-beta_2)*(g_t*g_t)	#updates the moving averages of the squared gradient (biased 2nd
                                                # raw moment estimate)

        m_cap = m_t / (1 - (beta_1 ** t))  # Compute bias-corrected first moment estimate
        v_cap = v_t / (1 - (beta_2 ** t))  # Compute bias-corrected second raw moment estimate
        X_t_prev = X_t
        X_t = X_t_prev - (learning_rate * m_cap) / (np.sqrt(v_cap) + epsilon)  # updates the parameters

        if disp is True:
            output = function(X_t)
            print('step: {} input:{} obj_funct: {}'.format(t, X_t, output))

        if np.isclose(X_t, X_t_prev, atol=tolerance).all(): # convergence check
            break
    if store_steps is True:
        return X_t, input_vectors, output_results
    else:
        return X_t

if __name__ == '__main__':
    def Function_to_minimise(input_vect, const=2):
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
        return np.array([dz_dx, dz_dy])

    X0 = np.array([1,2])
    GG = Adam_Opt(X0, calc_grad,
                  learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8)


    print(Function_to_minimise(GG))

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



### for scipy ###


# (fun, x0, args=args, jac=jac, hess=hess, hessp=hessp,
#                       bounds=bounds, constraints=constraints,
#                       callback=callback, **options)

def fmin_ADAM(f, x0, fprime=None, args=(), gtol=1e-5,
              maxiter=500, full_output=0, disp=1, maxfev=500,
              retall=0, callback=None, learning_rate = 0.001,
             beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8):
    """
    Minimize a function using the BFGS algorithm.
    Parameters
    ----------
    f : callable f(x,*args)
        Objective function to be minimized.
    x0 : ndarray
        Initial guess.
    delta (float): stepsize to approximate gradient
    """
    opts = {'gtol': gtol,
            'disp': disp,
            'maxiter': maxiter,
            'return_all': retall}


    res = _adam_minimize(f, x0, fprime, args=args, callback=callback,
                     xtol=gtol, maxiter=maxiter,
                     disp=disp, maxfev=maxfev, return_all=retall,
                   learning_rate = learning_rate,
                   beta_1 = beta_1, beta_2 = beta_2, epsilon=epsilon, **opts)

    if full_output:
        retlist = (res['x'], res['fun'], #res['jac'],
                   res['nfev'], res['status'])
        if retall:
            retlist += (res['allvecs'], )
        return retlist
    else:
        if retall:
            return res['x'], res['allvecs']
        else:
            return res['x']

    return result


from scipy.optimize.optimize import OptimizeResult, wrap_function, _status_message, _check_unknown_options


from numpy import squeeze
# _minimize_powell


def _adam_minimize(func, x0, args=(), jac=None, callback=None,
                     xtol=1e-8, maxiter=None, maxfev=None,
                     disp=False, return_all=False,
                   learning_rate = 0.001,
                   beta_1=0.9, beta_2=0.999, epsilon=1e-8, **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    modified Powell algorithm.
    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    xtol : float
        Relative error in solution `xopt` acceptable for convergence.
    ftol : float
        Relative error in ``fun(xopt)`` acceptable for convergence.
    maxiter, maxfev : int
        Maximum allowed number of iterations and function evaluations.
        Will default to ``N*1000``, where ``N`` is the number of
        variables, if neither `maxiter` or `maxfev` is set. If both
        `maxiter` and `maxfev` are set, minimization will stop at the
        first reached.
    direc : ndarray
        Initial set of direction vectors for the Powell method.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    """

    _check_unknown_options(unknown_options)

    if jac is None:
        raise ValueError('Jacobian is required for Adam-CG method')

    if maxfev is None:
        maxfev = maxiter + 10


    _, func = wrap_function(func, args)

    retall = return_all
    if retall:
        allvecs = [x0]
        all_jac_vecs=[jac(x0)]

    fval = squeeze(func(x0))

    # initialization
    t=0  # timestep
    m_t = 0 #1st moment vector
    v_t = 0 #2nd moment vector
    X_t = x0

    fcalls=0
    iter = 0
    while True:

        # ADAM Algorithm
        t+=1
        g_t = jac(X_t)
        m_t = beta_1*m_t + (1-beta_1)*g_t	#updates the moving averages of the gradient (biased first moment estimate)
        v_t = beta_2*v_t + (1-beta_2)*(g_t*g_t)	#updates the moving averages of the squared gradient (biased 2nd
                                                # raw moment estimate)

        m_cap = m_t / (1 - (beta_1 ** t))  # Compute bias-corrected first moment estimate
        v_cap = v_t / (1 - (beta_2 ** t))  # Compute bias-corrected second raw moment estimate
        X_t_prev = X_t
        X_t = X_t_prev - (learning_rate * m_cap) / (np.sqrt(v_cap) + epsilon)  # updates the parameters
        # Adam END

        # updates and termination criteria
        fcalls+=1
        fval = func(X_t)

        iter += 1
        if callback is not None:
            callback(X_t)
        if retall:
            allvecs.append(X_t)
            all_jac_vecs.append(g_t)

        if fcalls >= maxfev: # max function evaluation
            break
        if iter >= maxiter: # max no. of iterations
            break
        if np.isclose(X_t, X_t_prev, atol=xtol).all(): # convergence check
            break


    warnflag = 0
    if fcalls >= maxfev:
        warnflag = 1
        msg = _status_message['maxfev']
        if disp:
            print("Warning: " + msg)
    elif iter >= maxiter:
        warnflag = 2
        msg = _status_message['maxiter']
        if disp:
            print("Warning: " + msg)
    elif np.isnan(fval) or np.isnan(x).any():
        warnflag = 3
        msg = _status_message['nan']
        if disp:
            print("Warning: " + msg)
    else:
        msg = _status_message['success']
        if disp:
            print(msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % iter)
            print("         Function evaluations: %d" % fcalls)

    result = OptimizeResult(fun=fval, nit=iter, nfev=fcalls,
                            status=warnflag, success=(warnflag == 0),
                            message=msg, x=X_t)
    if retall:
        result['allvecs'] = allvecs
        result['jac'] = all_jac_vecs
    return result


if __name__ == '__main__':
    def Function_to_minimise(input_vect, const=2):
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
        return np.array([dz_dx, dz_dy])

    X0 = np.array([1,2])

    x = fmin_ADAM(Function_to_minimise, X0, fprime=calc_grad, learning_rate=1, maxiter=800, full_output=1, gtol=1e-5) #retall=1)
    print(x)

