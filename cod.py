from numpy import power
class COD:
    name = "Coefficient of determination (R^2)"
    shortname = "R^2"
    sign = -1

    def __init__(self, func, jac, hess, xdata, ydata):
        self.func = func
        self.fjac = jac
        self.fhess = hess
        self.data = list(zip(xdata, ydata))
        self.n = len(self.data)
        avg = sum(y for x, y in self.data) / self.n
        self.tot = 0.0
        for x, y in self.data:
            self.tot += power(y - avg, 2)
    
    def objfunc(self, params):
        res = 0.0
        for x, y in self.data:
            res += power(y - self.func(x, *params), 2)            
        return 1 - res / self.tot
    
    def jac(self, params):
        nominator = 0.0
        for x, y in self.data:
            nominator += (y - self.func(x, *params)) * self.fjac(x, *params)
        return 2 * nominator / self.tot
    
    def hess(self, params):
        pass
