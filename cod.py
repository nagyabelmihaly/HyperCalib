from numpy import power
class COD:
    name = "Coefficient of determination (R^2)"

    def __init__(self, func, jac, hess, xdata, ydata):
        self.func = func
        self.fjac = jac
        self.fhess = hess
        self.data = list(zip(xdata, ydata))
        self.n = len(self.data)
    
    def objfunc(self, params):
        res, tot = 0.0, 0.0
        avg = sum(x for x, y in self.data) / self.n
        for x, y in self.data:
            res += power(y - self.func(x, *params), 2)
            tot += power(y - avg, 2)
        return  res / tot
    
    def jac(self, params):
        pass
    
    def hess(self, params):
        pass
