from numpy import power, log, inf, array
from scipy.optimize import NonlinearConstraint

def ogden_ut(stretch, mu1, mu2, alpha1, alpha2):
    """Represents the K=2 Ogden model to uniaxial tension."""
    return mu1 * (power(stretch, alpha1 - 1) - power(stretch, -0.5 * alpha1 - 1)) + \
           mu2 * (power(stretch, alpha2 - 1) -  power(stretch, -0.5 * alpha2 - 1))

def ogden_et(stretch, mu1, mu2, alpha1, alpha2):
    """Represents the K=2 Ogden model to equibiaxial tension."""
    return mu1 * (power(stretch, alpha1 - 1) - power(stretch, -2 * alpha1 - 1)) + \
           mu2 * (power(stretch, alpha2 - 1) - power(stretch, -2 * alpha2 - 1))

def ogden_ps(stretch, mu1, mu2, alpha1, alpha2):
    """Represents the K=2 Ogden model to pure shear."""
    return mu1 * (power(stretch, alpha1 - 1) - power(stretch, -alpha1 - 1)) + \
           mu2 * (power(stretch, alpha2 - 1) - power(stretch, -alpha2 - 1))

def ogden_constraint():
    """Returns the constrain of the Ogden model."""
    def f(x):
        return [x[0] * x[2] + x[1] * x[3]]
    return NonlinearConstraint(f, 0, inf)

def ogden_ut_jac(stretch, mu1, mu2, alpha1, alpha2):
    """Returns the gradient vector of the K=2 Ogden model to uniaxial tension."""
    return array([power(stretch, alpha1 - 1) - power(stretch, -0.5 * alpha1 - 1),
            power(stretch, alpha2 - 1) - power(stretch, -0.5 * alpha2 - 1),
            mu1 * log(stretch) * (power(stretch, alpha1 - 1) + 0.5 * power(stretch, -0.5 * alpha1 - 1)),
            mu2 * log(stretch) * (power(stretch, alpha2 - 1) + 0.5 * power(stretch, -0.5 * alpha2 - 1))])

def ogden_et_jac(stretch, mu1, mu2, alpha1, alpha2):
    """Returns the gradient vector of the K=2 Ogden model to equibiaxial tension."""
    return array([power(stretch, alpha1 - 1) - power(stretch, -2 * alpha1 - 1),
            power(stretch, alpha2 - 1) - power(stretch, -2 * alpha2 - 1),
            mu1 * log(stretch) * (power(stretch, alpha1 - 1) + 2 * power(stretch, -2 * alpha1 - 1)),
            mu2 * log(stretch) * (power(stretch, alpha2 - 1) + 2 * power(stretch, -2 * alpha2 - 1))])

def ogden_ps_jac(stretch, mu1, mu2, alpha1, alpha2):
    """Returns the gradient vector of the K=2 Ogden model to pure shear."""
    return array([power(stretch, alpha1 - 1) - power(stretch, -alpha1 - 1),
            power(stretch, alpha2 - 1) - power(stretch, -alpha2 - 1),
            mu1 * log(stretch) * (power(stretch, alpha1 - 1) + power(stretch, -alpha1 - 1)),
            mu2 * log(stretch) * (power(stretch, alpha2 - 1) + power(stretch, -alpha2 - 1))])
