from ogden import Ogden
from errorfuncs import MSE

model = Ogden()
xdata = [1, 2]
ydata = [3, 4]
error = MSE(model.ps, model.ps_jac, model.ps_hess, xdata, ydata)
print(error.hess([3.525, 0.2873, 8.952, 2.0597]))