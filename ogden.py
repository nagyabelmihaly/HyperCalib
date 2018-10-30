from numpy import power, log, inf, array, empty, sqrt
from scipy.optimize import NonlinearConstraint

class Ogden:
    """Represents the Ogden model."""

    def __init__(self, n):
        self.n = n
        self.name = "Ogden N=" + str(n)
        self.paramnames = ["mu" + str(k + 1) for k in range(n)] + \
                          ["alpha" + str(k + 1) for k in range(n)]
        self.paramnames_latex = ['\\mu_{}'.format(str(k+1)) for k in range(n)] + \
                                ['\\alpha_{}'.format(str(k+1)) for k in range(n)]
        self.paramcount = len(self.paramnames)

        self.jacm = empty(2 * n)
        self.hessm = empty((2 * n, 2 * n))

    def f(self, stretch, c, params):
        n = self.n
        mu = params[:n]
        alpha = params[n:]
        return 2 * sum([mu[k] / alpha[k] * (power(stretch, alpha[k]) - power(stretch, -c * alpha[k])) for k in range(n)])

    def jac(self, stretch, c, params):
        n = self.n
        mu = params[:n]
        alpha = params[n:]

        for k in range(n):
            self.jacm[k] = 2 / alpha[k] * (power(stretch, alpha[k]) - power(stretch, -c * alpha[k]))
            self.jacm[n + k] = 2 * mu[k] * (-power(alpha[k], -2) * (power(stretch, alpha[k]) - power(stretch, -c * alpha[k])) +\
                log(stretch) / alpha[k] * (power(stretch, alpha[k]) + c * power(stretch, -c * alpha[k])))

    def hess(self, stretch, c, params):
        raise NotImplementedError()
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

    def func(self, defmode, stretch, *params):
        if defmode == 0:
            return self.ut(stretch, *params)
        if defmode == 1:
            return self.et(stretch, *params)
        if defmode == 2:
            return self.ps(stretch, *params)
        raise NotImplementedError

    def getfunc(self, defmode):
        if defmode == 0:
            return self.ut
        if defmode == 1:
            return self.et
        if defmode == 2:
            return self.ps
        raise NotImplementedError

    def getjac(self, defmode):
        if defmode == 0:
            return self.ut_jac
        if defmode == 1:
            return self.et_jac
        if defmode == 2:
            return self.ps_jac
        raise NotImplementedError

    def gethess(self, defmode):
        if defmode == 0:
            return self.ut_hess
        if defmode == 1:
            return self.et_hess
        if defmode == 2:
            return self.ps_hess
        raise NotImplementedError

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