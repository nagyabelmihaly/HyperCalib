from numpy import power, inf, array
from scipy.optimize import LinearConstraint

class MooneyRivlin:
    """Represents the Mooney-Rivlin model."""

    def __init__(self):
        self.name = "Mooney-Rivlin"
        self.paramnames = ["c10", "c01"]
        self.paramcount = len(self.paramnames)

    def ut(self, stretch, c10, c01):
        """Represents the Mooney-Rivlin model to uniaxial tension."""
        return c10 * (2 * stretch - 2 * power(stretch, -2)) + c01 * (2 - 2 * power(stretch, -3))

    def et(self, stretch, c10, c01):
        """Represents the Mooney-Rivlin model to equibiaxial tension."""
        return c10 * (2 * stretch - 2 * power(stretch, -5)) + c01 * (2 * power(stretch, 3) - 2 * power(stretch, -3))

    def ps(self, stretch, c10, c01):
        """Represents the Mooney-Rivlin model to pure shear."""
        return c10 * (2 * stretch - 2 * power(stretch, -3)) + c01 * (2 * stretch - 2 * power(stretch, -3))

    def constraint(self):
        """Returns the constrain of the Mooney-Rivlin model."""
        return LinearConstraint([[1, 1]], [0], [inf])

    def ut_jac(self, stretch, c10, c01):
        """Returns the gradient vector of the Mooney-Rivlin model to uniaxial tension."""
        return array([2 * stretch - 2 * power(stretch, -2),
                      2 - 2 * power(stretch, -3)])

    def et_jac(self, stretch, c10, c01):
        """Returns the gradient vector of the Mooney-Rivlin model to equibiaxial tension."""
        return array([2 * stretch - 2 * power(stretch, -5),
                      2 * power(stretch, 3) - 2 * power(stretch, -3)])

    def ps_jac(self, stretch, c10, c01):
        """Returns the gradient vector of the Mooney-Rivlin model to pure shear."""
        return array([2 * stretch - 2 * power(stretch, -3),
                      2 * stretch - 2 * power(stretch, -3)])

    def ut_hess(self, stretch, c10, c01):
        """Returns the Hessian matrix of the Mooney-Rivlin model to uniaxial tension."""
        return array([[0.0]])

    def et_hess(self, stretch, c10, c01):
        """Returns the Hessian matrix of the Mooney-Rivlin model to equibiaxial tension."""
        return array([[0.0]])

    def ps_hess(self, stretch, c10, c01):
        """Returns the Hessian matrix of the Mooney-Rivlin model to pure shear."""
        return array([[0.0]])
