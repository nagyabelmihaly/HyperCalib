from numpy import array, dot, empty

class MSE:
    def __init__(self, func, jac, hess, xdata, ydata):
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
        self.fhess = hess
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
            result -= (y - self.func(x, *params)) * self.fjac(x, *params)
        return result * 2 / len(self.data)
    
    def hess(self, params):
        """Calculates the Hessian matrix of the objective function
        when the parameters are applied."""
        return array([[2 / len(self.data) * sum([self.jac(params)[i] * self.jac(params)[j] - \
            (y - self.func(x, *params)) * self.fhess(x, *params)[i][j] for x, y in self.data]) \
            for i in range(len(params))] for j in range(len(params))])

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
        # result = empty((len(params), len(params)))
        # for i in range(len(params)):
        #     for j in range(len(params)):
        #         s = 0
        #         for error, weight in self.factors:
        #             s += error.hess(params)[i, j] * weight
        #         result[i, j] = s
        # return result
        return array([[sum(error.hess(params)[i, j] * weight \
            for error, weight in self.factors) \
            for i in range(len(params))] for j in range(len(params))])
    
    def hessp(self, params, p):
        """"""
        return self.hess(params).dot(p)
        