import math
import matplotlib.pyplot as plt
import numpy as np
import function

def derivative(f, x, dx = 1e-6):
    fpx = (f(x+dx) - f(x))/dx
    return fpx

#on cherche le 0 d'une fonction
def newton_raphson(f, x0, lim=1e-10, maxit=1000):
    x = x0
    fx = f(x)

    if abs(fx) < lim:
        return x

    for i in range(maxit):

        fpx = derivative(f,x)
        x = x - fx/fpx
        fx = f(x)

        if abs(fx) < lim:
            break
    
    return x

def compute_partial_derivative(f, x, axe_derive, h = 1e-6):
    x_new = x.copy()
    x_new[axe_derive] = x[axe_derive] + h
    print(f(x_new))
    print(f(x))
    fpx = (f(x_new) - f(x))/h
    return fpx

def compute_jacobian(f,x,dx = 1e-6):
    jacobian = np.zeros((len(f), len(x)))
    for i, fi in enumerate(f):
        for j, xj in enumerate(x):
            jacobian[i,j] = compute_partial_derivative(fi,x,j)
    return jacobian


def newton_raphson_ndim(x0, f, lim=1e-10, maxit=1000): #x0 as vector and f as vector
    x = x0
    fx = np.empty(len(f))
    for i, fi in enumerate(f):
        fx[i] = fi(x)

    if np.all(abs(fx) < lim): #If f(x) < lim is True for all functions
        return x

    for i in range(maxit):

        j = compute_jacobian(f,x)
        j_inv = np.linalg.pinv(j) #pinv = pseudo inverse matrix
        x = x - np.dot(j_inv,fx)
        for z, fz in enumerate(f):
            fx[z] = fz(x)
        if np.all(abs(fx) < lim): #If f(x) < lim is True for all functions
            break
    
    return x,fx

def nelder_mead(x_start, f):
    '''
    inputs : x_start as inputs with shape(x_start) = (d+1,d)
             f : function R^d -> R
    ouputs : 
    '''
    dim = np.shape(x_start)[1]
    print(dim)
    simplex = []
    for i in range(dim+1):
        simplex.append([np.array(x_start[i]),f(x_start[i])])
    
    
    while True:
        simplex.sort(key=lambda x: x[1])
        #barycentre X0
        X0 = [0.0]*dim
        for tup in simplex[:-1]: #without worst point
            for i, c in enumerate(tup[0]):
                X0[i] += c / (dim)

        #reflexion
        Xr = X0 + (X0-simplex[-1][0])
        f_Xr = f(Xr)

        #Etirement
        if f_Xr < f(simplex[0][0]):
            Xe = X0 + 2*(X0-simplex[-1][0])
            f_Xe = f(Xe)
            if f_Xe < f_Xr:
                simplex[-1][0] = Xe
                simplex[-1][1] = f_Xe
            else:
                simplex[-1][0] = Xr
                simplex[-1][1] = f_Xr
        
        elif f_Xr < f(simplex[-2][0]):
            simplex[-1][0] = Xr
            simplex[-1][1] = f_Xr
        
        #Contraction
        elif f_Xr >= f(simplex[-2][0]):
            #Contraction intérieur
            if f_Xr < f(simplex[-1][0]):
                Xc = X0 + 0.5*(X0 + Xr)
                f_Xc = f(Xc)
                if f_Xc < f_Xr:
                    simplex[-1][0] = Xc
                    simplex[-1][1] = f_Xc
            #Contraction extérieur
            elif f_Xr >= f(simplex[-1][0]):
                Xc = X0 + 0.5*(X0 + simplex[-1][0])
            
            

    #Contration
   

    return simplex
    

def print_3Dfunc(func):
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = func(np.array([X,Y]))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='terrain', edgecolor=None)
    ax.set(xlabel="x", ylabel="y", zlabel="f(x, y)", title="Surface")
    plt.show()

def print_2Dfunc(func):
    x = np.linspace(-10, 10, 100)
    y = func(x)
    plt.plot(x,y)
    plt.title("f(x) in function of x")
    plt.xlim([0, 1])
    plt.show()
 
if __name__ == "__main__":
    
    #x0 = 0.1
    #x = newton_raphson(function.BS_asian_call, x0)    
    #print('The solution is x = {}, f(x) = {}'.format(x,function.BS_asian_call(x))) 
    #print_2Dfunc(function.BS_asian_call)

    # myfuncs = np.array([function.func7])
    # x0 = np.array([3,3], dtype='float')
    # x, fx = newton_raphson_ndim(x0,myfuncs)
    # print("x = {}, y = {} pour f(x,y) = {}".format(x[0], x[1], fx))
    # print_3Dfunc(function.func7)

    x_start = np.array([[-1,1],[1,1],[0,0]],dtype='float')
    print(nelder_mead(x_start, function.func7))
