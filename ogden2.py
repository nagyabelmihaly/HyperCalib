from numpy import power, log, inf, array
from scipy.optimize import NonlinearConstraint

class Ogden2:
    """Represents the K=2 Ogden model."""

    def __init__(self):
        self.name = "Ogden K=2"
        self.paramnames = ["mu1", "mu2", "alpha1", "alpha2"]
        self.paramcount = len(self.paramnames)

    def ut(self, stretch, mu1, mu2, alpha1, alpha2):
        """Represents the K=2 Ogden model to uniaxial tension."""
        return mu1 * (power(stretch, alpha1 - 1) - power(stretch, -0.5 * alpha1 - 1)) + \
            mu2 * (power(stretch, alpha2 - 1) -  power(stretch, -0.5 * alpha2 - 1))

    def et(self, stretch, mu1, mu2, alpha1, alpha2):
        """Represents the K=2 Ogden model to equibiaxial tension."""
        return mu1 * (power(stretch, alpha1 - 1) - power(stretch, -2 * alpha1 - 1)) + \
            mu2 * (power(stretch, alpha2 - 1) - power(stretch, -2 * alpha2 - 1))

    def ps(self, stretch, mu1, mu2, alpha1, alpha2):
        """Represents the K=2 Ogden model to pure shear."""
        return mu1 * (power(stretch, alpha1 - 1) - power(stretch, -alpha1 - 1)) + \
            mu2 * (power(stretch, alpha2 - 1) - power(stretch, -alpha2 - 1))

    def constraint(self):
        """Returns the constrain of the K=2 Ogden model."""
        def f(x):
            return [x[0] * x[2] + x[1] * x[3]]
        return NonlinearConstraint(f, 0, inf)

    def ut_jac(self, stretch, mu1, mu2, alpha1, alpha2):
        """Returns the gradient vector of the K=2 Ogden model to uniaxial tension."""
        return array([power(stretch, alpha1 - 1) - power(stretch, -0.5 * alpha1 - 1),
                power(stretch, alpha2 - 1) - power(stretch, -0.5 * alpha2 - 1),
                mu1 * log(stretch) * (power(stretch, alpha1 - 1) + 0.5 * power(stretch, -0.5 * alpha1 - 1)),
                mu2 * log(stretch) * (power(stretch, alpha2 - 1) + 0.5 * power(stretch, -0.5 * alpha2 - 1))])

    def et_jac(self, stretch, mu1, mu2, alpha1, alpha2):
        """Returns the gradient vector of the K=2 Ogden model to equibiaxial tension."""
        return array([power(stretch, alpha1 - 1) - power(stretch, -2 * alpha1 - 1),
                power(stretch, alpha2 - 1) - power(stretch, -2 * alpha2 - 1),
                mu1 * log(stretch) * (power(stretch, alpha1 - 1) + 2 * power(stretch, -2 * alpha1 - 1)),
                mu2 * log(stretch) * (power(stretch, alpha2 - 1) + 2 * power(stretch, -2 * alpha2 - 1))])

    def ps_jac(self, stretch, mu1, mu2, alpha1, alpha2):
        """Returns the gradient vector of the K=2 Ogden model to pure shear."""
        return array([power(stretch, alpha1 - 1) - power(stretch, -alpha1 - 1),
                power(stretch, alpha2 - 1) - power(stretch, -alpha2 - 1),
                mu1 * log(stretch) * (power(stretch, alpha1 - 1) + power(stretch, -alpha1 - 1)),
                mu2 * log(stretch) * (power(stretch, alpha2 - 1) + power(stretch, -alpha2 - 1))])

    def ut_hess(self, stretch, mu1, mu2, alpha1, alpha2):
        """Returns the Hessian matrix of the K=2 Ogden model to uniaxial tension."""
        mu1alpha1 = log(stretch) * (power(stretch, alpha1 - 1) + 0.5 * power(stretch, -0.5 * alpha1 - 1))
        mu2alpha2 = log(stretch) * (power(stretch, alpha2 - 1) + 0.5 * power(stretch, -0.5 * alpha2 - 1))
        alpha1alpha1 = mu1 * power(log(stretch), 2) * (power(stretch, alpha1 - 1) - 0.25 * power(stretch, -0.5 * alpha1 - 1))
        alpha2alpha2 = mu2 * power(log(stretch), 2) * (power(stretch, alpha2 - 1) - 0.25 * power(stretch, -0.5 * alpha2 - 1))
        return array([[0.0, 0.0, mu1alpha1, 0.0],
                    [0.0, 0.0, 0.0, mu2alpha2],
                    [mu1alpha1, 0.0, alpha1alpha1, 0.0],
                    [0.0, mu2alpha2, 0.0, alpha2alpha2]])

    def et_hess(self, stretch, mu1, mu2, alpha1, alpha2):
        """Returns the Hessian matrix of the K=2 Ogden model to equibiaxial tension."""
        mu1alpha1 = log(stretch) * (power(stretch, alpha1 - 1) + 2 * power(stretch, -2 * alpha1 - 1))
        mu2alpha2 = log(stretch) * (power(stretch, alpha2 - 1) + 2 * power(stretch, -2 * alpha2 - 1))
        alpha1alpha1 = mu1 * power(log(stretch), 2) * (power(stretch, alpha1 - 1) - 4 * power(stretch, -2 * alpha1 - 1))
        alpha2alpha2 = mu2 * power(log(stretch), 2) * (power(stretch, alpha2 - 1) - 4 * power(stretch, -2 * alpha2 - 1))
        return array([[0.0, 0.0, mu1alpha1, 0.0],
                    [0.0, 0.0, 0.0, mu2alpha2],
                    [mu1alpha1, 0.0, alpha1alpha1, 0.0],
                    [0.0, mu2alpha2, 0.0, alpha2alpha2]])

    def ps_hess(self, stretch, mu1, mu2, alpha1, alpha2):
        """Returns the Hessian matrix of the K=2 Ogden model to pure shear."""
        mu1alpha1 = log(stretch) * (power(stretch, alpha1 - 1) + power(stretch, -alpha1 - 1))
        mu2alpha2 = log(stretch) * (power(stretch, alpha2 - 1) + power(stretch, -alpha2 - 1))
        alpha1alpha1 = mu1 * power(log(stretch), 2) * (power(stretch, alpha1 - 1) - power(stretch, -alpha1 - 1))
        alpha2alpha2 = mu2 * power(log(stretch), 2) * (power(stretch, alpha2 - 1) - power(stretch, -alpha2 - 1))
        return array([[0.0, 0.0, mu1alpha1, 0.0],
                    [0.0, 0.0, 0.0, mu2alpha2],
                    [mu1alpha1, 0.0, alpha1alpha1, 0.0],
                    [0.0, mu2alpha2, 0.0, alpha2alpha2]])