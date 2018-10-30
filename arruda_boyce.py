from numpy import power, array, zeros, sqrt, abs,\
    tan, sign, spacing, size

class ArrudaBoyce:
    """Represents the Arruda-Boyce model."""

    def __init__(self):
        self.name = "Arruda-Boyce"
        self.paramnames = ["mu", "lambda_lock"]
        self.paramnames_latex = ['\mu', '\lambda_{lock}']
        self.paramcount = len(self.paramnames)
    
        self.a = 1.31446
        self.b = 1.58986
        self.c = 0.91209
        self.d = 0.84136

    def lambda_chain_ut(self, stretch):
        i1 = power(stretch, 2) + 2 / stretch
        return sqrt(i1 / 3)

    def lambda_chain_et(self, stretch):
         i1 = 2 * power(stretch, 2) + power(stretch, -4)
         return sqrt(i1 / 3)

    def lambda_chain_ps(self, stretch):
        i1 = power(stretch, 2) + 1 + power(stretch, -2)
        return sqrt(i1 / 3)

    def inv_langevin(self, x):
        eps = spacing(1)
        # x is a scalar
        if isinstance(x, float):
            if x >= 1 - eps:
                x = 1 - eps
            if x <= -1 + eps:
                x = -1 + eps
            if abs(x) < self.d:
                return self.a * tan(self.b * x) + self.c * x
            return 1.0 / (sign(x) - x)
        # x is an array
        x[x >= 1 - eps] = 1 - eps
        x[x <= -1 + eps] = -1 + eps
        res = zeros(size(x))
        index = abs(x) < self.d
        res[index] = self.a * tan(self.b * x[index]) + self.c * x[index]
        index = abs(x) >= self.d
        res[index] = 1.0 / (sign(x[index]) - x[index])
        return res

    def right(self, stretch, c):
        return power(stretch, 2) - power(stretch, -c)

    def f(self, stretch, c, lambda_chain, mu, lambda_lock):
        return mu / lambda_chain * self.inv_langevin(lambda_chain / lambda_lock) / self.inv_langevin(1 / lambda_lock) * self.right(stretch, c)

    def jac(self, stretch, c, i1):
        raise NotImplementedError
        right = self.right(stretch, c)
        return array([2 * right,
                4 * (i1 - 3) * right,
                6 * power(i1 - 3, 2) * right])

    def constraint(self, x):
        """Returns the constrain of the Arruda-Boyce model."""
        return x[0]

    def func(self, defmode, stretch, mu, lambda_lock):
        if defmode == 0:
            return self.ut(stretch, mu, lambda_lock)
        if defmode == 1:
            return self.et(stretch, mu, lambda_lock)
        if defmode == 2:
            return self.ps(stretch, mu, lambda_lock)
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

    def ut(self, stretch, mu, lambda_lock):
        """Represents the Arruda-Boyce model to uniaxial tension."""
        return self.f(stretch, 1, self.lambda_chain_ut(stretch), mu, lambda_lock)

    def et(self, stretch, mu, lambda_lock):
        """Represents the Arruda-Boyce model to equibiaxial tension."""
        return self.f(stretch, 4, self.lambda_chain_et(stretch), mu, lambda_lock)

    def ps(self, stretch, mu, lambda_lock):
        """Represents the Arruda-Boyce model to pure shear."""
        return self.f(stretch, 2, self.lambda_chain_ps(stretch), mu, lambda_lock)

    def ut_jac(self, stretch, mu, lambda_lock):
        """Returns the gradient vector of the Arruda-Boyce model to uniaxial tension."""
        raise NotImplementedError
        return self.jac(stretch, 1, self.i1ut(stretch))

    def et_jac(self, stretch, mu, lambda_lock):
        """Returns the gradient vector of the Arruda-Boyce model to equibiaxial tension."""
        raise NotImplementedError
        return self.jac(stretch, 4, self.i1et(stretch))

    def ps_jac(self, stretch, mu, lambda_lock):
        """Returns the gradient vector of the Arruda-Boyce model to pure shear."""
        raise NotImplementedError
        return self.jac(stretch, 2, self.i1ps(stretch))

    def ut_hess(self, stretch, mu, lambda_lock):
        """Returns the Hessian matrix of the Arruda-Boyce model to uniaxial tension."""
        raise NotImplementedError
        z = zeros((3, 3))
        return z

    def et_hess(self, stretch, mu, lambda_lock):
        """Returns the Hessian matrix of the Arruda-Boyce model to equibiaxial tension."""
        raise NotImplementedError
        return zeros((3, 3))

    def ps_hess(self, stretch, mu, lambda_lock):
        """Returns the Hessian matrix of the Arruda-Boyce model to pure shear."""
        raise NotImplementedError
        return zeros((3, 3))


