import numpy as np
from scipy.stats import norm


def BlackNScholesClosedFormula(sigma, K, r, T, S0):
    if (K == 0):
        return S0
    d1 = (1 / (sigma * np.sqrt(T))) * (np.log(S0 / K) + (r + (1 / 2) * sigma ** 2) * T)
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)



#print(BlackNScholesClosedFormula(sigma, K, r, T, S0))

#This pricer seems pretty good
