from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.optimize import BFGS

from errorfuncs import MSE

def fit_model(objfunc, constraints=(), callback=None):
    """Finds the optimal parameters for the given function and input data.
    ----------
    Keyword arguments:
    objfunc -- The objective (error) function.
               Should have syntax objfunc(params) -> error.
    constraints -- A single (or a list of) constrain objects. (default: ())
    callback -- This callback is called after each iteration step.
                Should have syntax callback(xk, OptimizeResult state)
                where xk is the current parameter vector. (default: ())
    """
    params = [1.0] * 4
    res = minimize(objfunc, params, method='trust-constr', \
                   constraints=constraints, \
                   jac='2-point', hess=BFGS(),
                   callback=callback,
                   options={'maxiter':10000, 'disp': True})
    return res.x
