from numpy import power, inf, array, zeros
from scipy.optimize import LinearConstraint

class MooneyRivlin:
    """Represents the Mooney-Rivlin model."""

    def __init__(self):
        self.name = "Mooney-Rivlin"
        self.paramnames = ["c10", "c01"]
        self.paramcount = len(self.paramnames)

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
