import numpy as np
from scipy.optimize import NonlinearConstraint

def ogden_ut(stretch, mu1, mu2, alpha1, alpha2):
    return mu1 * (np.power(stretch, alpha1 - 1) - np.power(stretch, -0.5 * alpha1 - 1)) + \
           mu2 * (np.power(stretch, alpha2 - 1) -  np.power(stretch, -0.5 * alpha2 - 1))

def ogden_et(stretch, mu1, mu2, alpha1, alpha2):
    return mu1 * (np.power(stretch, alpha1 - 1) - np.power(stretch, -2 * alpha1 - 1)) + \
           mu2 * (np.power(stretch, alpha2 - 1) - np.power(stretch, -2 * alpha2 - 1))

def ogden_ps(stretch, mu1, mu2, alpha1, alpha2):
    return mu1 * (np.power(stretch, alpha1 - 1) - np.power(stretch, -alpha1 - 1)) + \
           mu2 * (np.power(stretch, alpha2 - 1) - np.power(stretch, -alpha2 - 1))

def ogden_constraints():
    def f(x):
        return [x[0] * x[2] + x[1] * x[3]]
    return NonlinearConstraint(f, 0, np.inf)