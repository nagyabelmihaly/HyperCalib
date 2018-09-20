from numpy import power, array

class NeoHooke:
    """Represents the Neo-Hooke model."""
    
    def __init__(self):
        self.name = "Neo-Hooke"
        self.paramnames = ["mu"]
        self.paramcount = len(self.paramnames)

    def ut(self, stretch, mu):
        """Represents the Neo-Hooke model to uniaxial tension."""
        return mu * (stretch - power(stretch, -2))

    def et(self, stretch, mu):
        """Represents the Neo-Hooke model to equibiaxial tension."""
        return mu * (stretch - power(stretch, -5))

    def ps(self, stretch, mu):
        """Represents the Neo-Hooke model to pure shear."""
        return mu * (stretch - power(stretch, -3))

    def constraint(self):
        """Returns the constrain of the Neo-Hooke model."""
        return ()

    def ut_jac(self, stretch, mu):
        """Returns the gradient vector of the Neo-Hooke model to uniaxial tension."""
        return array([stretch - power(stretch, -2)])

    def et_jac(self, stretch, mu):
        """Returns the gradient vector of the Neo-Hooke model to equibiaxial tension."""
        return array([stretch - power(stretch, -5)])

    def ps_jac(self, stretch, mu):
        """Returns the gradient vector of the Neo-Hooke model to pure shear."""
        return array([stretch - power(stretch, -3)])

    def ut_hess(self, stretch, mu):
        """Returns the Hessian matrix of the Neo-Hooke model to uniaxial tension."""
        return array([[0.0]])

    def et_hess(self, stretch, mu):
        """Returns the Hessian matrix of the Neo-Hooke model to equibiaxial tension."""
        return array([[0.0]])

    def ps_hess(self, stretch, mu):
        """Returns the Hessian matrix of the Neo-Hooke model to pure shear."""
        return array([[0.0]])
