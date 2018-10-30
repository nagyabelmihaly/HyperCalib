from numpy import power, array

class NeoHooke:
    """Represents the Neo-Hooke model."""
    
    def __init__(self):
        self.name = "Neo-Hooke"
        self.paramnames = ["mu"]
        self.paramnames_latex = ['\mu']
        self.paramcount = len(self.paramnames)

    def constraint(self, x):
        """Returns the constrain of the Neo-Hooke model."""
        return 1

    def func(self, defmode, stretch, mu):
        if defmode == 0:
            return self.ut(stretch, mu)
        if defmode == 1:
            return self.et(stretch, mu)
        if defmode == 2:
            return self.ps(stretch, mu)
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

    def ut(self, stretch, mu):
        """Represents the Neo-Hooke model to uniaxial tension."""
        return mu * (power(stretch, 2) - power(stretch, -1))

    def et(self, stretch, mu):
        """Represents the Neo-Hooke model to equibiaxial tension."""
        return mu * (power(stretch, 2) - power(stretch, -4))

    def ps(self, stretch, mu):
        """Represents the Neo-Hooke model to pure shear."""
        return mu * (power(stretch, 2) - power(stretch, -2))

    def ut_jac(self, stretch, mu):
        """Returns the gradient vector of the Neo-Hooke model to uniaxial tension."""
        return array([power(stretch, 2) - power(stretch, -1)])

    def et_jac(self, stretch, mu):
        """Returns the gradient vector of the Neo-Hooke model to equibiaxial tension."""
        return array([power(stretch, 2) - power(stretch, -4)])

    def ps_jac(self, stretch, mu):
        """Returns the gradient vector of the Neo-Hooke model to pure shear."""
        return array([power(stretch, 2) - power(stretch, -2)])

    def ut_hess(self, stretch, mu):
        """Returns the Hessian matrix of the Neo-Hooke model to uniaxial tension."""
        return array([[0.0]])

    def et_hess(self, stretch, mu):
        """Returns the Hessian matrix of the Neo-Hooke model to equibiaxial tension."""
        return array([[0.0]])

    def ps_hess(self, stretch, mu):
        """Returns the Hessian matrix of the Neo-Hooke model to pure shear."""
        return array([[0.0]])
