from numpy import array, dot, empty, sqrt

class MSRE:
    name = "Mean Squared Relative Error"

    def __init__(self, func, jac, hess, xdata, ydata):
        self.func = func
        self.fjac = jac
        self.fhess = hess
        self.data = list(zip(xdata, ydata))

    def objfunc(self, params):
        error = 0
        for x, y in self.data:
            if y == 0:
                continue
            error += ((self.func(x, *params) - y) / y) ** 2
        return sqrt(error / len(self.data))

    def jac(self, params):
        raise NotImplementedError()
        result = array([0.0] * len(params))
        for x, y in self.data:
            result += 2 / x ** 2 * (self.func(x, *params) - y) * self.fjac(x, *params)
        return result / len(self.data)

    def hess(self, params):
        raise NotImplementedError()
        return array([[2 / len(self.data) * sum([1 / x ** 2 * (self.fjac(x, *params)[i] * self.fjac(x, *params)[j] + \
            (self.func(x, *params) - y) * self.fhess(x, *params)[i][j] for x, y in self.data)]) \
            for i in range(len(params))] for j in range(len(params))])