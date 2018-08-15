from numpy import array

class MSE:
    def __init__(self, func, jac, xdata, ydata):
        """Initializes a MSE instance whose objective function
        calculates the mean squared error between the given
        data points and the function.
        ----------
        Keyword arguments:
        func -- Function should have syntax func(x, *params).
        jac -- A callable calculating the gradient vector
               of the function.
        xdata -- The list of x coordinates of size (n,).
        ydata -- The list of y coordinates of size (n,).
        """
        self.func = func
        self.fjac = jac
        self.data = list(zip(xdata, ydata))
    
    def objfunc(self, params):
        """Returns the MSE when the parameters are applied."""
        error = 0
        for x, y in self.data:
            error += (y - self.func(x, *params)) ** 2
        return error / len(self.data)
    
    def jac(self, params):
        """Calculates the gradient vector of the objective function
        when the parameters are applied."""
        result = array([0.0] * len(params))
        for x, y in self.data:
            diff = y - self.func(x, *params)
            j = self.fjac(x, *params)
            inc = diff * j
            result += inc
        return result * 2 / len(self.data)

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
            j = err.jac(params)
            inc = j * weight
            result += inc
        return result