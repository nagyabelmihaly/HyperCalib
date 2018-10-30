from numpy import array, zeros, sqrt, matmul, transpose

class RMSAE:
    name = "Root Mean Squared Absolute Error"
    shortname = "RMSAE"
    name_latex = '\\Delta_{RMSA}'

    def __init__(self, func, jac, hess, xdata, ydata):
        """Initializes a RRMSAE instance whose objective function
        calculates the square root of the mean squared error
        between the given data points and the function.
        ----------
        Keyword arguments:
        func -- Function should have syntax func(x, *params).
        jac -- A callable calculating the gradient vector
               of the function.
        hess -- A callable calculating the Hessian matrix
                of the function.
        xdata -- The list of x coordinates of size (n,).
        ydata -- The list of y coordinates of size (n,).
        """
        self.func = func
        self.fjac = jac
        self.fhess = hess

        length = len(xdata)
        not_origin_index = [y != 0.0 for y in ydata]
        self.data = list(zip([xdata[i] for i in range(length) if not_origin_index[i]],
                             [ydata[i] for i in range(length) if not_origin_index[i]]))
        self.n = len(self.data)
    
    def objfunc(self, params):
        """Returns the RMSAE when the parameters are applied."""
        error = 0
        for x, y in self.data:
            f = self.func(x, *params)
            abserr = f - y
            error += abserr ** 2
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
        r = result / (self.n * objf) - matmul(jacobian, transpose(jacobian)) / objf
        return r
