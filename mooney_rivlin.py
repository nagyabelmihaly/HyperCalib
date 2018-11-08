from numpy import power, inf, array, zeros
from scipy.optimize import LinearConstraint

class MooneyRivlin:
    """Represents the Mooney-Rivlin model."""

    def __init__(self):
        self.name = "Mooney-Rivlin"
        self.paramnames = ["C10", "C01"]
        self.paramnames_latex = ["C_{10}", "C_{01}"]
        self.paramcount = len(self.paramnames)

    def func(self, defmode, stretch, c10, c01):
        if defmode == 0:
            return self.ut(stretch, c10, c01)
        if defmode == 1:
            return self.et(stretch, c10, c01)
        if defmode == 2:
            return self.ps(stretch, c10, c01)
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

    def ut(self, stretch, c10, c01):
        """Represents the Mooney-Rivlin model to uniaxial tension."""
        return 2 * (power(stretch, 2) - 1 / stretch) * (c10 + c01 / stretch)

    def et(self, stretch, c10, c01):
        """Represents the Mooney-Rivlin model to equibiaxial tension."""
        return 2 * c10 * (power(stretch, 2) - power(stretch, -4)) + \
               2 * c01 * (power(stretch, 4) - power(stretch, -2))

    def ps(self, stretch, c10, c01):
        """Represents the Mooney-Rivlin model to pure shear."""
        return 2 * (power(stretch, 2) - power(stretch, -2)) * (c10 + c01)

    def constraint(self, x):
        """Returns the constrain of the Mooney-Rivlin model."""
        return x[0] + x[1]

    def guess(self):
        return [1.0, 1.0]

    def ut_jac(self, stretch, c10, c01):
        """Returns the gradient vector of the Mooney-Rivlin model to uniaxial tension."""
        return array([2 * (power(stretch, 2) - 1 / stretch),
                      2 * (stretch - power(stretch, -2))])

    def et_jac(self, stretch, c10, c01):
        """Returns the gradient vector of the Mooney-Rivlin model to equibiaxial tension."""
        return array([2 * (power(stretch, 2) - power(stretch, -4)),
                      2 * (power(stretch, 4) - power(stretch, -2))])

    def ps_jac(self, stretch, c10, c01):
        """Returns the gradient vector of the Mooney-Rivlin model to pure shear."""
        return array([2 * (power(stretch, 2) - power(stretch, -2)),
                      2 * (power(stretch, 2) - power(stretch, -2))])

    def ut_hess(self, stretch, c10, c01):
        """Returns the Hessian matrix of the Mooney-Rivlin model to uniaxial tension."""
        return zeros((2, 2))

    def et_hess(self, stretch, c10, c01):
        """Returns the Hessian matrix of the Mooney-Rivlin model to equibiaxial tension."""
        return zeros((2, 2))

    def ps_hess(self, stretch, c10, c01):
        """Returns the Hessian matrix of the Mooney-Rivlin model to pure shear."""
        return zeros((2, 2))
