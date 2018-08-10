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

def fit_model(func, xdata, ydata, constraints):
    ef = ErrorFunc(func, xdata, ydata)
    params = [1.0] * 4
    res = minimize(ef.objfunc, params, method='trust-constr', \
                   constraints=constraints, tol=1e-8, \
                   jac='2-point', hess=BFGS(),
                   options={'maxiter':10000, 'disp': True})
    return res.x
