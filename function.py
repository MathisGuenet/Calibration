import math
import numpy as np
from scipy import stats


func1 = lambda x : math.exp(x) - 2 
func2 = lambda x : x**3 + 2  
func3 = lambda x : math.log(x)
func4 = lambda x : math.atan(x)+2*(math.cos(math.cos(x))-1)
func5 = lambda x : 1 - math.exp(-x**2)
func6 = lambda x : math.sqrt(abs(x))
func7 = lambda x : (x[0]**2)*x[1]+2
func8 = lambda x : np.log(1 + (x[0]**2 + x[1]**2 - 1)**2)
func9 = lambda x : 20 + math.e - 20 * math.exp(-0.2 * math.sqrt((x[0]**2 + x[1]**2) / 1)) - math.exp((math.cos(2 * math.pi * x[0]) + math.cos(2 * math.pi * x[1]) / 1))
func10 = lambda x : x**2
func11 = lambda x : math.sqrt(abs(x))
func12 = lambda x : (x[0]**2) + (x[1]**2) +(x[2]**2)
func13 = lambda x : -np.cos(np.pi*x[0])*np.sin(np.pi*x[1])*np.exp(-(x[0]**2 + x[1]**2)/10)
Styblinsky_Tang = lambda x : 0.5*((x[0]**4 + x[1]**4) - 16*(x[0]**2+x[1]**2) + 5*(x[0]+x[1]))

def BS_european_call(sigma):
    S = 100
    K = 101
    r = 0.01
    T = 5
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    call = (S * stats.norm.cdf(d1, 0.0, 1.0) - 
            K * np.exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0))
    market_price = 19.38 #Market price with 0.20 implied volatility
    return abs(call - market_price)

def BS_asian_call(sigma):
    Nprix = 5
    Nsimul = 10000
    maturity = 5
    S = 100
    K = 101
    r = 0.01

    #compute n paths of Nprix (=maturity)
    w = np.random.normal(0,1,(Nsimul, Nprix))
    delta_t = maturity/Nprix 
    var = (r-(sigma**2)/2)*delta_t + sigma*np.sqrt(delta_t)*w
    var = np.cumsum(var,axis = 1)
    St = S*np.exp(var)
    mean_paths = np.mean(St, axis = 1)
    payoffs = np.maximum((mean_paths) - K, 0)
    price = np.mean(payoffs)
    #print(price)
    return price - 12.77
