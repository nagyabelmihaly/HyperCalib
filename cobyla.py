from scipy.optimize import minimize

class Cobyla:
    name = "COBYLA"
    support_jac = False
    support_hess = False
    support_xtol = True
    support_gtol = False
    support_iterations = True
    support_rounds = True

    default_jac = False
    default_hess = False
    default_xtol = '1e-8'
    default_gtol = ''
    default_iterations = '1000'
    default_rounds = '10'

    def get_constraint(self, fun):
        return {'type': 'ineq', 'fun': fun}

    def minimize(self, objfunc, x0, constraint, jac, hess, callback, xtol, gtol, maxiter, rounds):
        nfev = 0
        for i in range(rounds):
            result = minimize(objfunc, x0,
                     method='COBYLA',
                     constraints=self.get_constraint(constraint),
                     tol=xtol,
                     options={'maxiter': maxiter, 'disp': True})
            nfev += result.nfev
            result.niter = 0
            x0 = result.x
            callback(x0, result)
        result.nfev = nfev
        result.niter = nfev
        return result