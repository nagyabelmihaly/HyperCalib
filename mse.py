from numpy import array, zeros, sqrt, matmul, transpose

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
        self.n = len(self.data)
    
    def objfunc(self, params):
        """Returns the MSE when the parameters are applied."""
        error = 0
        for x, y in self.data:
            error += (self.func(x, *params) - y) ** 2
        return sqrt(error / self.n)
    
    def jac(self, params):
        """Calculates the gradient vector of the objective function
        when the parameters are applied."""
        result = zeros(len(params))
        for x, y in self.data:
            result += (self.func(x, *params) - y) * self.fjac(x, *params)
        return result / (self.n * self.objfunc(params))
    
    def hess(self, params):
        """Calculates the Hessian matrix of the objective function
        when the parameters are applied."""
        result = zeros((len(params), len(params)))
        for x, y in self.data:
            fj = self.fjac(x, *params)
            result += matmul(fj, transpose(fj)) + (self.func(x, *params) - y) * self.fhess(x, *params)
        jacobian = self.jac(params)
        objf = self.objfunc(params)
        return result / (self.n * objf) - matmul(jacobian, transpose(jacobian)) / objf
