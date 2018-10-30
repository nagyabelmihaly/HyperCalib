from numpy import power, array, zeros

class Yeoh:
    """Represents the Yeoh model."""
    
    def __init__(self):
        self.name = "Yeoh"
        self.paramnames = ["C10", "C20", "C30"]
        self.paramnames_latex = ['C_{10}', 'C_{20}', 'C_{30}']
        self.paramcount = len(self.paramnames)

    def i1ut(self, stretch):
        return power(stretch, 2) + 2 / stretch

    def i1et(self, stretch):
        return 2 * power(stretch, 2) + power(stretch, -4)

    def i1ps(self, stretch):
        return power(stretch, 2) + 1 + power(stretch, -2)

    def a(self, stretch, c):
        return power(stretch, 2) - power(stretch, -c)

    def f(self, stretch, c, i1, c10, c20, c30):
        return 2 * (c10 + 2 * c20 * (i1 - 3) + 3 * c30 * power(i1 - 3, 2)) * self.a(stretch, c)

    def jac(self, stretch, c, i1):
        a = self.a(stretch, c)
        return array([2 * a,
                4 * (i1 - 3) * a,
                6 * power(i1 - 3, 2) * a])

    def constraint(self, x):
        """Returns the constrain of the Yeoh model."""
        return x[0]

    def func(self, defmode, stretch, c10, c20, c30):
        if defmode == 0:
            return self.ut(stretch, c10, c20, c30)
        if defmode == 1:
            return self.et(stretch, c10, c20, c30)
        if defmode == 2:
            return self.ps(stretch, c10, c20, c30)
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

    def ut(self, stretch, c10, c20, c30):
        """Represents the Yeoh model to uniaxial tension."""
        return self.f(stretch, 1, self.i1ut(stretch), c10, c20, c30)

    def et(self, stretch, c10, c20, c30):
        """Represents the Yeoh model to equibiaxial tension."""
        return self.f(stretch, 4, self.i1et(stretch), c10, c20, c30)

    def ps(self, stretch, c10, c20, c30):
        """Represents the Yeoh model to pure shear."""
        return self.f(stretch, 2, self.i1ps(stretch), c10, c20, c30)

    def ut_jac(self, stretch, c10, c20, c30):
        """Returns the gradient vector of the Yeoh model to uniaxial tension."""
        return self.jac(stretch, 1, self.i1ut(stretch))

    def et_jac(self, stretch, c10, c20, c30):
        """Returns the gradient vector of the Yeoh model to equibiaxial tension."""
        return self.jac(stretch, 4, self.i1et(stretch))

    def ps_jac(self, stretch, c10, c20, c30):
        """Returns the gradient vector of the Yeoh model to pure shear."""
        return self.jac(stretch, 2, self.i1ps(stretch))

    def ut_hess(self, stretch, c10, c20, c30):
        """Returns the Hessian matrix of the Yeoh model to uniaxial tension."""
        z = zeros((3, 3))
        return z

    def et_hess(self, stretch, c10, c20, c30):
        """Returns the Hessian matrix of the Yeoh model to equibiaxial tension."""
        return zeros((3, 3))

    def ps_hess(self, stretch, c10, c20, c30):
        """Returns the Hessian matrix of the Yeoh model to pure shear."""
        return zeros((3, 3))

