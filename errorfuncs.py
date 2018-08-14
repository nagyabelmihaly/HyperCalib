class MSE:
    def __init__(self, func, xdata, ydata):
        """Initializes a MSE instance whose objective function
        calculates the mean squared error between the given
        data points and the function.
        ----------
        Keyword arguments:
        func -- Function should have syntax func(x, *params).
        xdata -- The list of x coordinates of size (n,).
        ydata -- The list of y coordinates of size (n,).
        """
        self.func = func
        self.data = list(zip(xdata, ydata))
    
    def objfunc(self, params):
        """Returns the MSE when the parameters are applied."""
        error = 0
        for x, y in self.data:
            error += (y - self.func(x, *params)) ** 2 / len(self.data)
        return error

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