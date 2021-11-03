import matplotlib.pyplot as plt
import numpy as np
import function

def plot_3Dfunc(func, title = None):
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = func(np.array([X,Y]))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='terrain', edgecolor=None)
    ax.set(xlabel="x", ylabel="y", zlabel="f(x, y)", title="Surface")
    if title is None:
        plt.title("f(x) in function of x")
    else:
        plt.title(title)
    plt.show()

def plot_2Dfunc(func, title = None):
    x = np.linspace(-10, 10, 1000)
    y = func(x)
    plt.plot(x,y)
    if title is None:
        plt.title("f(x) in function of x")
    else:
        plt.title(title)
    plt.xlim([0, 0.4])
    plt.show()