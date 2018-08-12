from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.optimize import BFGS

class ErrorFunc:
    def __init__(self, func, xdata, ydata):
        self.func = func
        self.xdata = xdata
        self.ydata = ydata
    
    def objfunc(self, params):
        result = 0
        for x, y in zip(self.xdata, self.ydata):
            result += (y - self.func(x, *params)) ** 2
        return result

def fit_model(func, xdata, ydata, constraints=(), callback=None):
    """Finds the optimal parameters for the given function and input data.
    ----------
    Keyword arguments:
    func -- The function whose parameters are to be optimized.
            Should have syntax func(x, *params).
    xdata -- The list of x coordinates of size (n,).
    ydata -- The list of y coordinates of size (n,).
    constraints -- A single (or a list of) constrain objects. (default: ())
    callback -- This callback is called after each iteration step.
                Should have syntax callback(xk, OptimizeResult state)
                where xk is the current parameter vector.
    """
    ef = ErrorFunc(func, xdata, ydata)
    params = [1.0] * 4
    res = minimize(ef.objfunc, params, method='trust-constr', \
                   constraints=constraints, \
                   jac='2-point', hess=BFGS(),
                   callback=callback,
                   options={'maxiter':10000, 'disp': True})
    return res.x
