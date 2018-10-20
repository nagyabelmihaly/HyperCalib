from numpy import inf
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize

class TrustConstr:
    name = "trust-constr"

    def __init__(self):
        self.xtol = 1e-8
        self.gtol = 1e-8
        self.maxiter = 1000

    def print_params(self):
        jac_string = "Calculate Jacobian" if self.calcJac else "Estimate Jacobian"
        hess_string = "Calculate Hessian" if self.calcHess else "Estimate Hessian"
        return """{0}
{1}
Xtol: {2}
Gtol: {3}
Maximum iterations: {4}""".format(jac_string, hess_string, self.xtol, self.gtol, self.maxiter)

    def get_constraint(self):
        return NonlinearConstraint(self.constraint, 0, inf)

    def minimize(self, callback):
        result = minimize(self.objfunc, self.x0,
                        method='trust-constr',
                        constraints=self.get_constraint(),
                        jac=self.jac, hess=self.hess,
                        callback=callback,
                        options={'xtol': self.xtol, 'gtol': self.gtol,
                                 'maxiter': self.maxiter, 'disp': True})
        result.print = """Optimality: {0:.4g}
Maximum constraint violation: {1:.4g}
Number of iterations: {2}
Number of function evaluations: {3}
Number of Jacobian evaluations: {4}
Number of Hessian evaluations: {5}
Number of conjugate gradient method iterations: {6}
Radius of the trust region at the last iteration: {7:.4g}
Penalty parameter at the last iteration: {8:.4g}
Barrier tolerance: {9:.4g}
Barrier parameter: {10:.4g}
{11}""".format(result.optimality, result.constr_violation,
              result.niter, result.nfev, result.njev, result.nhev, result.cg_niter,
              result.tr_radius, result.constr_penalty, result.barrier_tolerance,
              result.barrier_parameter, result.message)
        return result
