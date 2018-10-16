from scipy.optimize import minimize, OptimizeResult

class Slsqp:
    name = "SLSQP"
    support_jac = True
    support_hess = False
    support_xtol = True
    support_gtol = True
    support_iterations = True
    support_rounds = True

    default_jac = True
    default_hess = False
    default_xtol = '1e-8'
    default_gtol = '1e-6'
    default_iterations = '1000'
    default_rounds = '10'

    def get_constraint(self, fun):
        return {'type': 'ineq', 'fun': fun}

    def minimize(self, objfunc, x0, constraint, jac, hess, callback, xtol, gtol, maxiter, rounds):
        self.fobjfunc = objfunc
        self.fcallback = callback
        self.nfev = 0
        for i in range(rounds):
            result = minimize(objfunc, x0,
                     method='SLSQP',
                     constraints=self.get_constraint(constraint),
                     tol=xtol,
                     callback=self.callback,
                     options={'ftol': gtol, 'maxiter': maxiter, 'disp': True})
            self.nfev += result.nfev
            result.niter = 0
            x0 = result.x
            #callback(x0, result)
        result.nfev = self.nfev
        result.niter = self.nfev
        return result

    def callback(self, x):
        state = OptimizeResult()
        state.niter = self.nfev
        state.fun = self.fobjfunc(x)
        self.fcallback(x, state)
