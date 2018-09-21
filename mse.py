from numpy import array, dot, empty, sqrt

class MSE:
    name = "Mean Squared Error"

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
            error += (self.func(x, *params) - y) ** 2
        return sqrt(error / len(self.data))
    
    def jac(self, params):
        """Calculates the gradient vector of the objective function
        when the parameters are applied."""
        raise NotImplementedError()
        result = array([0.0] * len(params))
        for x, y in self.data:
            result += 2 * (self.func(x, *params) - y) * self.fjac(x, *params)
        return result / len(self.data)
    
    def hess(self, params):
        """Calculates the Hessian matrix of the objective function
        when the parameters are applied."""
        raise NotImplementedError()
        return array([[2 / len(self.data) * sum([self.fjac(x, *params)[i] * self.fjac(x, *params)[j] + \
            (self.func(x, *params) - y) * self.fhess(x, *params)[i][j] for x, y in self.data]) \
            for i in range(len(params))] for j in range(len(params))])
