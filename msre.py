from numpy import array, zeros, sqrt, matmul, transpose

class MSRE:
    name = "Mean Squared Relative Error"

    def __init__(self, func, jac, hess, xdata, ydata):
        self.func = func
        self.fjac = jac
        self.fhess = hess
        self.data = list(zip(xdata, ydata))
        self.n = len(self.data)

    def objfunc(self, params):
        error = 0
        for x, y in self.data:
            error += ((self.func(x, *params) - y) / y) ** 2
        return sqrt(error / len(self.data))

    def jac(self, params):
        result = zeros(len(params))
        for x, y in self.data:
            result += (self.func(x, *params) - y) / y ** 2 * self.fjac(x, *params)
        return result / (self.n * self.objfunc(params))

    def hess(self, params):
        result = zeros((len(params), len(params)))
        for x, y in self.data:
            fj = self.fjac(x, *params)
            result += (matmul(fj, transpose(fj)) + (self.func(x, *params) - y) * self.fhess(x, *params)) / y ** 2
        jacobian = self.jac(params)
        objf = self.objfunc(params)
        return result / (self.n * objf) - matmul(jacobian, transpose(jacobian)) / objf