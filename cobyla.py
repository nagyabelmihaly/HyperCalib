from scipy.optimize import minimize

class Cobyla:
    name = "COBYLA"

    def __init__(self):
        self.tol = None
        self.rhobeg = 1.0
        self.maxiter = 1000
        self.catol = 0.0002

    def print_params(self):
        return """Tol: {0}
Rhobeg: {1}
Maximum iterations: {2}
Catol: {3}""".format(self.tol, self.rhobeg, self.maxiter, self.catol)


    def get_constraint(self):
        return {'type': 'ineq', 'fun': self.constraint}

    def minimize(self, callback):
        result = minimize(self.objfunc, self.x0,
                        method='COBYLA',
                        constraints=self.get_constraint(),
                        tol=self.tol,
                        callback=callback,
                        options={'rhobeg': self.rhobeg,
                                 'maxiter': self.maxiter,
                                 'disp': True,
                                 'catol': self.catol})
        result.niter = 1
        result.print = """Number of function evaluations: {0}
{1} {2}""".format(result.nfev,
                 '' if result.success else 'Optimization failed.',
                 result.message)
        return result