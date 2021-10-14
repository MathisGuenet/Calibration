import numpy as np
import matplotlib.pyplot as plt 
from scipy import stats


def BS_MonteCarlo(S0, K, maturity, r, sigma, Nsimul, option):
    """
    BS dynamic : St = S0*exp((r-(sigma^2)/2)*T + sigma*T)
    we want to simule Nsimul price with a given maturity
    we dont calcul all the path, only the price at maturity
    """
    Nprix = 20
    #Generate Matrix of standard normal random variable
    w = np.random.normal(0,1,(Nsimul, Nprix))
    #compute delta_t
    delta_t = maturity/Nprix
    #Compute the Matrix of Nprix variation for all simulation 
    var = (r-(sigma**2)/2)*delta_t + sigma*np.sqrt(delta_t)*w
    #Compute the sum of variation for each simulation, return a vector
    var = np.cumsum(var,axis = 1)
    St = S0*np.exp(var)
    if option == "EuropeanCall":
        #Compute the vector of all final price's simulation
        #compute the payoff for each simulation
        #compute the discounted mean to get the final price of the call 
        ST = St[:,-1]
        price = np.mean(np.maximum(ST-K,0))*np.exp(-r*maturity)
    elif option == "EuropeanPut":
        #Compute the vector of all final price's simulation
        #compute the payoff for each simulation
        #compute the mean payoffs
        #compute the discounted mean to get the final price of the put
        ST = St[:,-1]
        price = np.mean(np.maximum(K-ST,0))*np.exp(-r*maturity)
    elif option == "AsianCall":
        meanPrice = np.mean(St, axis = 1)
        payoff = np.maximum(meanPrice-K, 0)
        price = np.mean(payoff)*np.exp(-r*maturity)
    elif option == "AsianPut":
        meanPrice = np.mean(St, axis = 1)
        payoff = np.maximum(K-meanPrice, 0)
        price = np.mean(payoff)*np.exp(-r*maturity)
    return price, St

def BS_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    call = (S * stats.norm.cdf(d1, 0.0, 1.0) - 
            K * np.exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0))
    return call

def BS_delta(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return stats.norm.cdf(d1, 0.0, 1.0)

def BS_gamma(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return stats.norm.pdf(d1, 0.0, 1.0)/(S*sigma*np.sqrt(T))

def printPrices(all_prices, option_price, maturity):
    nb_of_prices = len(all_prices[0])
    indexes = np.linspace(0, maturity, nb_of_prices)

    plt.axis([indexes.min(), indexes.max(), all_prices.min() - 1, all_prices.max() + 1])
    plt.xlabel("Temps.")
    plt.ylabel("Valeur de l'option.")
    plt.title("Estimation de prix selon la méthode de Monte Carlo.")

    plt.plot([], color='w', label='Pricing :' + str(round(option_price, 2)) + "€")

    plt.plot(indexes.T, all_prices.T)

    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    europeanCall, pricesEcall = BS_MonteCarlo(105,100,2,0.01,0.10,1000, "EuropeanCall")
    europeanPut, pricesEput  = BS_MonteCarlo(100,100,2,0.01,0.15,1000, "EuropeanPut")
    asianCall, pricesAcall = BS_MonteCarlo(100,100,2,0.01,0.15,1000, "AsianCall")
    asianPut, pricesAput = BS_MonteCarlo(100,100,2,0.01,0.15,1000, "AsianPut")
    print(europeanCall)
    print(europeanPut)
    print(asianCall)
    print(asianPut)
    printPrices(pricesEcall, europeanCall, 2)
