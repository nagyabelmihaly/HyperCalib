from numpy import power
class COD:
    name = "Coefficient of determination (R^2)"
    shortname = "R^2"

    def __init__(self, func, jac, hess, xdata, ydata):
        self.func = func
        self.fjac = jac
        self.fhess = hess
        
        length = len(xdata)
        not_origin_index = [y != 0.0 for y in ydata]
        self.data = list(zip([xdata[i] for i in range(length) if not_origin_index[i]],
                             [ydata[i] for i in range(length) if not_origin_index[i]]))
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
