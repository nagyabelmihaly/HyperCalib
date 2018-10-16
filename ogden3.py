from numpy import power, log, inf, array
from scipy.optimize import NonlinearConstraint

class Ogden3:
    """Represents the K=3 Ogden model."""

    def __init__(self):
        self.name = "Ogden K=3"
        self.paramnames = ["mu1", "mu2", "mu3", "alpha1", "alpha2", "alpha3"]
        self.paramcount = len(self.paramnames)

    def ut(self, stretch, mu1, mu2, mu3, alpha1, alpha2, alpha3):
        """Represents the K=3 Ogden model to uniaxial tension."""
        return mu1 * (power(stretch, alpha1 - 1) - power(stretch, -0.5 * alpha1 - 1)) + \
            mu2 * (power(stretch, alpha2 - 1) -  power(stretch, -0.5 * alpha2 - 1)) + \
            mu3 * (power(stretch, alpha3 - 1) -  power(stretch, -0.5 * alpha3 - 1))

    def et(self, stretch, mu1, mu2, mu3, alpha1, alpha2, alpha3):
        """Represents the K=3 Ogden model to equibiaxial tension."""
        return mu1 * (power(stretch, alpha1 - 1) - power(stretch, -2 * alpha1 - 1)) + \
            mu2 * (power(stretch, alpha2 - 1) - power(stretch, -2 * alpha2 - 1)) + \
            mu3 * (power(stretch, alpha3 - 1) -  power(stretch, -2 * alpha3 - 1))

    def ps(self, stretch, mu1, mu2, mu3, alpha1, alpha2, alpha3):
        """Represents the K=3 Ogden model to pure shear."""
        return mu1 * (power(stretch, alpha1 - 1) - power(stretch, -alpha1 - 1)) + \
            mu2 * (power(stretch, alpha2 - 1) - power(stretch, -alpha2 - 1)) + \
            mu3 * (power(stretch, alpha3 - 1) -  power(stretch, -alpha3 - 1))

    def constraint(self, x):
        """Returns the constrain of the K=3 Ogden model."""
        return [x[0] * x[3] + x[1] * x[4] + x[2] * x[5]]

    def ut_jac(self, stretch, mu1, mu2, mu3, alpha1, alpha2, alpha3):
        """Returns the gradient vector of the K=3 Ogden model to uniaxial tension."""
        return array([power(stretch, alpha1 - 1) - power(stretch, -0.5 * alpha1 - 1),
                power(stretch, alpha2 - 1) - power(stretch, -0.5 * alpha2 - 1),
                power(stretch, alpha3 - 1) - power(stretch, -0.5 * alpha3 - 1),
                mu1 * log(stretch) * (power(stretch, alpha1 - 1) + 0.5 * power(stretch, -0.5 * alpha1 - 1)),
                mu2 * log(stretch) * (power(stretch, alpha2 - 1) + 0.5 * power(stretch, -0.5 * alpha2 - 1)),
                mu3 * log(stretch) * (power(stretch, alpha3 - 1) + 0.5 * power(stretch, -0.5 * alpha3 - 1))])

    def et_jac(self, stretch, mu1, mu2, mu3, alpha1, alpha2, alpha3):
        """Returns the gradient vector of the K=3 Ogden model to equibiaxial tension."""
        return array([power(stretch, alpha1 - 1) - power(stretch, -2 * alpha1 - 1),
                power(stretch, alpha2 - 1) - power(stretch, -2 * alpha2 - 1),
                power(stretch, alpha3 - 1) - power(stretch, -2 * alpha3 - 1),
                mu1 * log(stretch) * (power(stretch, alpha1 - 1) + 2 * power(stretch, -2 * alpha1 - 1)),
                mu2 * log(stretch) * (power(stretch, alpha2 - 1) + 2 * power(stretch, -2 * alpha2 - 1)),
                mu3 * log(stretch) * (power(stretch, alpha3 - 1) + 2 * power(stretch, -2 * alpha3 - 1))])

    def ps_jac(self, stretch, mu1, mu2, mu3, alpha1, alpha2, alpha3):
        """Returns the gradient vector of the K=3 Ogden model to pure shear."""
        return array([power(stretch, alpha1 - 1) - power(stretch, -alpha1 - 1),
                power(stretch, alpha2 - 1) - power(stretch, -alpha2 - 1),
                power(stretch, alpha3 - 1) - power(stretch, -alpha3 - 1),
                mu1 * log(stretch) * (power(stretch, alpha1 - 1) + power(stretch, -alpha1 - 1)),
                mu2 * log(stretch) * (power(stretch, alpha2 - 1) + power(stretch, -alpha2 - 1)),
                mu3 * log(stretch) * (power(stretch, alpha3 - 1) + power(stretch, -alpha3 - 1))])

    def ut_hess(self, stretch, mu1, mu2, mu3, alpha1, alpha2, alpha3):
        """Returns the Hessian matrix of the K=3 Ogden model to uniaxial tension."""
        mu1alpha1 = log(stretch) * (power(stretch, alpha1 - 1) + 0.5 * power(stretch, -0.5 * alpha1 - 1))
        mu2alpha2 = log(stretch) * (power(stretch, alpha2 - 1) + 0.5 * power(stretch, -0.5 * alpha2 - 1))
        mu3alpha3 = log(stretch) * (power(stretch, alpha3 - 1) + 0.5 * power(stretch, -0.5 * alpha3 - 1))
        alpha1alpha1 = mu1 * power(log(stretch), 2) * (power(stretch, alpha1 - 1) - 0.25 * power(stretch, -0.5 * alpha1 - 1))
        alpha2alpha2 = mu2 * power(log(stretch), 2) * (power(stretch, alpha2 - 1) - 0.25 * power(stretch, -0.5 * alpha2 - 1))
        alpha3alpha3 = mu3 * power(log(stretch), 2) * (power(stretch, alpha3 - 1) - 0.25 * power(stretch, -0.5 * alpha3 - 1))
        return array([[0.0, 0.0, 0.0, mu1alpha1, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, mu2alpha2, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, mu3alpha3],
                    [mu1alpha1, 0.0, 0.0, alpha1alpha1, 0.0, 0.0],
                    [0.0, mu2alpha2, 0.0, 0.0, alpha2alpha2, 0.0],
                    [0.0, 0.0, mu3alpha3, 0.0, 0.0, alpha3alpha3]])

    def et_hess(self, stretch, mu1, mu2, mu3, alpha1, alpha2, alpha3):
        """Returns the Hessian matrix of the K=3 Ogden model to equibiaxial tension."""
        mu1alpha1 = log(stretch) * (power(stretch, alpha1 - 1) + 2 * power(stretch, -2 * alpha1 - 1))
        mu2alpha2 = log(stretch) * (power(stretch, alpha2 - 1) + 2 * power(stretch, -2 * alpha2 - 1))
        mu3alpha3 = log(stretch) * (power(stretch, alpha3 - 1) + 2 * power(stretch, -2 * alpha3 - 1))
        alpha1alpha1 = mu1 * power(log(stretch), 2) * (power(stretch, alpha1 - 1) - 4 * power(stretch, -2 * alpha1 - 1))
        alpha2alpha2 = mu2 * power(log(stretch), 2) * (power(stretch, alpha2 - 1) - 4 * power(stretch, -2 * alpha2 - 1))
        alpha3alpha3 = mu3 * power(log(stretch), 2) * (power(stretch, alpha3 - 1) - 4 * power(stretch, -2 * alpha3 - 1))
        return array([[0.0, 0.0, 0.0, mu1alpha1, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, mu2alpha2, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, mu3alpha3],
                    [mu1alpha1, 0.0, 0.0, alpha1alpha1, 0.0, 0.0],
                    [0.0, mu2alpha2, 0.0, 0.0, alpha2alpha2, 0.0],
                    [0.0, 0.0, mu3alpha3, 0.0, 0.0, alpha3alpha3]])

    def ps_hess(self, stretch, mu1, mu2, mu3, alpha1, alpha2, alpha3):
        """Returns the Hessian matrix of the K=3 Ogden model to pure shear."""
        mu1alpha1 = log(stretch) * (power(stretch, alpha1 - 1) + power(stretch, -alpha1 - 1))
        mu2alpha2 = log(stretch) * (power(stretch, alpha2 - 1) + power(stretch, -alpha2 - 1))
        mu3alpha3 = log(stretch) * (power(stretch, alpha3 - 1) + power(stretch, -alpha3 - 1))
        alpha1alpha1 = mu1 * power(log(stretch), 2) * (power(stretch, alpha1 - 1) - power(stretch, -alpha1 - 1))
        alpha2alpha2 = mu2 * power(log(stretch), 2) * (power(stretch, alpha2 - 1) - power(stretch, -alpha2 - 1))
        alpha3alpha3 = mu3 * power(log(stretch), 2) * (power(stretch, alpha3 - 1) - power(stretch, -alpha3 - 1))
        return array([[0.0, 0.0, 0.0, mu1alpha1, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, mu2alpha2, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, mu3alpha3],
                    [mu1alpha1, 0.0, 0.0, alpha1alpha1, 0.0, 0.0],
                    [0.0, mu2alpha2, 0.0, 0.0, alpha2alpha2, 0.0],
                    [0.0, 0.0, mu3alpha3, 0.0, 0.0, alpha3alpha3]])