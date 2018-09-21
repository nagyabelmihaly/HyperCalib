from numpy import array, dot, empty, sqrt

class WeightedError:
    def __init__(self, errors, weights):
        """Initializes a WeightedError instance whose objective
        function calculates the weighted error of the given
        input errors.
        ----------
        Keyword arguments:
        errors -- The individual error instances.
        weights -- The importance factors of the errors.
        """
        self.factors = [(error, weight) for error, weight in zip(errors, weights) if error is not None]
    
    def objfunc(self, params):
        """Returns the weighted error when the parameters are applied."""
        error = 0
        for err, weight in self.factors:
            error += err.objfunc(params) * weight
        return error
    
    def jac(self, params):
        """Calculates the gradient vector of the objective function
        when the parameters are applied."""
        result = array([0.0] * len(params))
        for err, weight in self.factors:
            result += err.jac(params) * weight
        return result
    
    def hess(self, params):
        """Calculates the Hessian matrix of the objective function
        when the parameters are applied."""
        return array([[sum(error.hess(params)[i, j] * weight \
            for error, weight in self.factors) \
            for i in range(len(params))] for j in range(len(params))])
