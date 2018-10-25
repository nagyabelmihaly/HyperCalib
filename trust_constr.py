from numpy import inf
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize

class TrustConstr:
    name = "trust-constr"

    def __init__(self):
        self.xtol = 1e-8
        self.gtol = 1e-8
        self.barrier_tol = 1e-8
        self.maxiter = 1000
        self.initial_constr_penalty = 1.0
        self.initial_tr_radius = 1.0
        self.initial_barrier_parameter = 0.1
        self.initial_barrier_tolerance = 0.1

    def print_params(self):
        jac_string = "Calculate Jacobian" if self.calcJac else "Estimate Jacobian"
        hess_string = "Calculate Hessian" if self.calcHess else "Estimate Hessian"
        return """{0}
{1}
Xtol: {2}
Gtol: {3}
Barrier tolerance: {4}
Maximum iterations: {5}
Initial constrains penalty: {6}
Initial trust radius: {7}
Initial barrier parameter: {8}
Initial barrier tolerance: {9}""".format(jac_string, hess_string, self.xtol, self.gtol,
                                         self.barrier_tol, self.maxiter,
                                         self.initial_constr_penalty,
                                         self.initial_tr_radius,
                                         self.initial_barrier_parameter,
                                         self.initial_barrier_tolerance)

    def get_constraint(self):
        return NonlinearConstraint(self.constraint, 0, inf)

    def minimize(self, callback):
        result = minimize(self.objfunc, self.x0,
                        method='trust-constr',
                        constraints=self.get_constraint(),
                        jac=self.jac, hess=self.hess,
                        callback=callback,
                        options={'xtol': self.xtol, 'gtol': self.gtol,
                                 'barrier_tol': self.barrier_tol, 
                                 'maxiter': self.maxiter,
                                 'initial_constr_penalty': self.initial_constr_penalty,
                                 'initial_tr_radius': self.initial_tr_radius,
                                 'initial_barrier_parameter': self.initial_barrier_parameter,
                                 'initial_barrier_tolerance': self.initial_barrier_tolerance,
                                 'disp': True})
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
