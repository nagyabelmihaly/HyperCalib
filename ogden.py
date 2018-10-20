from numpy import power, log, inf, array, empty
from scipy.optimize import NonlinearConstraint

class Ogden:
    """Represents the Ogden model."""

    def __init__(self, n):
        self.n = n
        self.name = "Ogden N=" + str(n)
        self.paramnames = ["mu" + str(k + 1) for k in range(n)] + \
                          ["alpha" + str(k + 1) for k in range(n)]
        self.paramcount = len(self.paramnames)

        self.jacm = empty(2 * n)
        self.hessm = empty((2 * n, 2 * n))

    def f(self, stretch, c, params):
        n = self.n
        mu = params[:n]
        alpha = params[n:]
        return 2 * sum([mu[k] * (power(stretch, alpha[k] - 1) - power(stretch, -c * alpha[k] - 1)) for k in range(n)])

    def jac(self, stretch, c, params):
        n = self.n
        mu = params[:n]
        alpha = params[n:]

        for k in range(n):
            self.jacm[k] = 2 * (power(stretch, alpha[k] - 1) - power(stretch, -c * alpha[k] - 1))
            self.jacm[n + k] = 2 * mu[k] * log(stretch) * (power(stretch, alpha[k] - 1) + c * power(stretch, -c * alpha[k] - 1))

    def hess(self, stretch, c, params):
        n = self.n
        mu = params[:n]
        alpha = params[n:]
        
        for i in range(n):
            for j in range(n):
                self.hessm[i, j] = 0.0
                if i == j:
                    self.hessm[n + i, n + j] = 2 * mu[i] * power(log(stretch), 2) * \
                        (power(stretch, alpha[i] - 1) - power(c, 2) * power(stretch, -c * alpha[i] - 1))
                    self.hessm[n + i, j] = 2 * log(stretch) * \
                        (power(stretch, alpha[i] - 1) + c * power(stretch, -c * alpha[i] - 1))
                    self.hessm[i, n + j] = self.hessm[n + i, j]
                else:
                    self.hessm[n + i, n + j] = 0.0
                    self.hessm[n + i, j] = 0.0
                    self.hessm[i, n + j] = 0.0

    def constraint(self, x):
        """Returns the constrain of the Ogden model."""
        n = self.n
        return [sum([x[i] * x[n + i] for i in range(n)])]

    def ut(self, stretch, *params):
        """Represents the Ogden model to uniaxial tension."""
        return self.f(stretch, 0.5, params)

    def et(self, stretch, *params):
        """Represents the Ogden model to equibiaxial tension."""
        return self.f(stretch, 2.0, params)

    def ps(self, stretch, *params):
        """Represents the Ogden model to pure shear."""
        return self.f(stretch, 1.0, params)

    def ut_jac(self, stretch, *params):
        """Returns the gradient vector of the Ogden model to uniaxial tension."""
        self.jac(stretch, 0.5, params)
        return self.jacm

    def et_jac(self, stretch, *params):
        """Returns the gradient vector of the Ogden model to equibiaxial tension."""
        self.jac(stretch, 2.0, params)
        return self.jacm

    def ps_jac(self, stretch, *params):
        """Returns the gradient vector of the Ogden model to pure shear."""
        self.jac(stretch, 1.0, params)
        return self.jacm

    def ut_hess(self, stretch, *params):
        """Returns the Hessian matrix of the Ogden model to uniaxial tension."""
        self.hess(stretch, 0.5, params)
        return self.jacm

    def et_hess(self, stretch, *params):
        """Returns the Hessian matrix of the Ogden model to equibiaxial tension."""
        self.hess(stretch, 2.0, params)
        return self.jacm

    def ps_hess(self, stretch, *params):
        """Returns the Hessian matrix of the Ogden model to pure shear."""
        self.hess(stretch, 1.0, params)
        return self.jacm