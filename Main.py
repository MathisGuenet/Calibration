from Optimize import newton_raphson, newton_raphson_ndim, print_3Dfunc, print_2Dfunc
import function
import numpy as np

def optimize_newton_raphson():
    x0 = 0.1
    x = newton_raphson(function.BS_asian_call, x0)    
    print('The solution is x = {}, f(x) = {}'.format(x,function.BS_asian_call(x))) 
    print_2Dfunc(function.BS_asian_call)

def optimize_newton_rhapson_ndim(myfuncs):
    x0 = np.array([3,3], dtype='float')
    x, fx = newton_raphson_ndim(x0,myfuncs)
    print("x = {}, y = {} pour f(x,y) = {}".format(x[0], x[1], fx))
    print_3Dfunc(function.func7)

if __name__ == "__main__":
    myfuncs = np.array([function.func2])
    optimize_newton_rhapson_ndim(myfuncs)