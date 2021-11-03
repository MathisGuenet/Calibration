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

def dichotomie(x_start, f, lim=1e-10, **kwarg):
    '''
    Search for f(x) = 0
    inputs : x_start as input with len(x_start) = (2)
             x_start = [a,b]
             f : function R -> R
    output : 
    '''
    a = x_start[0]
    b = x_start[1]
    if f(a, **kwarg) * f(b, **kwarg) >= 0:
        print("Initialized parameters are wrong. f(a)*f(b) must be < 0")
        return x_start
    while abs(b - a) > lim:
        c = (a + b)/2
        fa = f(a, **kwarg)
        fc = f(c, **kwarg)
        if fc == 0:
            return c, fc
        elif fc*fa < 0:
            a = a
            b = c
        elif fc*fa > 0:
            a = c
            b = b
    return c, f(c, **kwarg)


def nelder_mead(x_start, f, max_iter = 1000, lim = 10e-6, **kwargs):
    '''
    Minimize the function f
    inputs : x_start as inputs with shape(x_start) = (d+1,d)
             f : function R^d -> R
             max_iter : number of maximum iteration
             lim : treshold of simplex volume
             **kwargs : parameters of the function f
    ouputs : simplex : [[x1,f(x1)], [x2,f(x2)], ...]
    '''
    dim = np.shape(x_start)[1]
    #print(dim)
    simplex = []
    for i in range(dim+1):
        simplex.append([np.array(x_start[i]),f(x_start[i], **kwargs)])
    
    iter = 0
    while iter < max_iter and np.all(np.abs(simplex[0][0] - simplex[-1][0]) < lim) == False:
        #print(np.abs(simplex[0][0] - simplex[-1][0]) < lim)
        iter += 1
        simplex.sort(key=lambda x: x[1])
        #barycentre X0
        X0 = [0.0]*dim
        for tup in simplex[:-1]: #without worst point
            for i, c in enumerate(tup[0]):
                X0[i] += c / (dim)

        #reflexion
        Xr = X0 + (X0-simplex[-1][0])
        f_Xr = f(Xr, **kwargs)

        #Etirement
        if f_Xr < simplex[0][1]:
            Xe = X0 + 2*(X0-simplex[-1][0])
            f_Xe = f(Xe, **kwargs)
            if f_Xe < f_Xr:
                simplex[-1][0] = Xe
                simplex[-1][1] = f_Xe
            else:
                simplex[-1][0] = Xr
                simplex[-1][1] = f_Xr
        
        elif f_Xr < simplex[-2][1]:
            simplex[-1][0] = Xr
            simplex[-1][1] = f_Xr
        
        #Contraction
        elif f_Xr >= simplex[-2][1]:
            #Contraction intérieur
            if f_Xr < simplex[-1][1]:
                Xc = 0.5*(X0 + Xr)
                f_Xc = f(Xc, **kwargs)
                if f_Xc <= f_Xr:
                    simplex[-1][0] = Xc
                    simplex[-1][1] = f_Xc
                #Xc pas bon candidat
                else: 
                    for i in range(1,len(simplex)):
                        simplex[i][0] = 0.5*(simplex[1][0] + simplex[i][0])
                        simplex[i][1] = f(simplex[i][0], **kwargs)
            #Contraction extérieur
            elif f_Xr >= simplex[-1][1]:
                Xc = 0.5*(X0 + simplex[-1][0])
                f_Xc = f(Xc, **kwargs)
                if f_Xc <= simplex[-1][1]:
                    simplex[-1][0] = Xc
                    simplex[-1][1] = f_Xc
                #Xc pas bon candidat
                else: 
                    for i in range(1,len(simplex)):
                        simplex[i][0] = 0.5*(simplex[1][0] + simplex[i][0])
                        simplex[i][1] = f(simplex[i][0], **kwargs)

    #print("number of itération = " + str(iter))
    simplex.sort(key=lambda x: x[1])
    return simplex

def nelder_mead_global(dimension, f, upperbound = 10, lowerbound = -10, **kwargs):
    res = []
    for i in range(dimension+1):
        x_start = np.random.uniform(lowerbound,upperbound, (dimension +1,dimension))
        res.append(nelder_mead(x_start, f, **kwargs)[0])
    x_start = []
    #print(res)
    for i in range(dimension+1):
        x_start.append(res[i][0])
    final_res = nelder_mead(x_start,f, **kwargs)
    return final_res[0]

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
    x = np.linspace(-10, 10, 1000)
    y = func(x)
    plt.plot(x,y)
    plt.title("f(x) in function of x")
    plt.xlim([0, 1])
    plt.show()
 
if __name__ == "__main__":
    #x_start = [-10,10]
    #print(dichotomie(x_start,function.func2))
    #print_2Dfunc(function.func2)

    #x_start = np.array([[0.5],[1]],dtype='float')
    #print_2Dfunc(function.BS_european_call)
    #print(nelder_mead(x_start, function.BS_european_call)[0])

    #print(nelder_mead_global(1,function.BS_european_call,0,10))
    print(nelder_mead_global(3,function.func12,-10,10))
    print(nelder_mead_global(2,function.func13,-50,50))
    print(nelder_mead_global(2,function.Styblinsky_Tang,-10,10))
    dict_param = {
        'T': 5,
        'S' : 100,
        'K' : 101,
        'r' : 0.01, 
        'market_price' : 19.38
    }
    print(nelder_mead_global(1, function.BS_european_call,0,1, **dict_param))
 


    
