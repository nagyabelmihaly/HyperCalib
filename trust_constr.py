from numpy import inf
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize

class TrustConstr:
    name = "trust-constr"

    support_jac = True
    support_hess = True
    support_xtol = True
    support_gtol = True
    support_iterations = True
    support_rounds = False

    default_jac = True
    default_hess = False
    default_xtol = '1e-8'
    default_gtol = '1e-8'
    default_iterations = '1000'
    default_rounds = ''

    def get_constraint(self, fun):
        return NonlinearConstraint(fun, 0, inf)

    def minimize(self, objfunc, x0, constraint, jac, hess, callback,
                 xtol, gtol, maxiter, rounds):
        return minimize(objfunc, x0,
                        method='trust-constr',
                        constraints=self.get_constraint(constraint),
                        jac=jac, hess=hess,
                        callback=callback,
                        options={'xtol': xtol, 'gtol': gtol,
                                 'maxiter': maxiter, 'disp': True})
