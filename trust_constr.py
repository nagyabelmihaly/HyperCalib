from numpy import inf
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize

class TrustConstr:
    name = "trust-constr"

    def __init__(self):
        self.xtol = 1e-8
        self.gtol = 1e-8
        self.maxiter = 1000

    def get_constraint(self, fun):
        return NonlinearConstraint(fun, 0, inf)

    def minimize(self, objfunc, x0, constraint, jac, hess, callback):
        return minimize(objfunc, x0,
                        method='trust-constr',
                        constraints=self.get_constraint(constraint),
                        jac=jac, hess=hess,
                        callback=callback,
                        options={'xtol': self.xtol, 'gtol': self.gtol,
                                 'maxiter': self.maxiter, 'disp': True})
