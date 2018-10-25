from scipy.optimize import minimize, OptimizeResult

class Slsqp:
    name = "SLSQP"

    def __init__(self):
        self.tol = None
        self.maxiter = 100
        self.ftol = 1e-6
        self.eps = 1.4901161193847656e-08

    def print_params(self):
        return """Tol: {0}
Maximum iterations: {1}
Ftol: {2}
Eps: {3}""".format(self.tol, self.maxiter, self.ftol, self.eps)

    def get_constraint(self):
        return {'type': 'ineq', 'fun': self.constraint}

    def minimize(self, callback):
        result = minimize(self.objfunc, self.x0,
                    method='SLSQP',
                    constraints=self.get_constraint(),
                    tol=self.tol, callback=callback,
                    options={'maxiter': self.maxiter,
                             'ftol': self.ftol,
                             'eps': self.eps,
                             'disp': True})
        result.niter = 1
        result.print = """Number of function evaluations: {0}
{1}{2}""".format(result.nfev,
                 '' if result.success else 'Optimization failed. ',
                 result.message)
        return result
