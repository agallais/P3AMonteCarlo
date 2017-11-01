import numpy as np
import time
import BlackNScholes_ClosedFormula as BS
import matplotlib.pyplot as plt


# nombre de simulations, correspond au parmêtre Ml

sigma = 0.2
r = 0.05
K = 100
T = 1.0
S0 = 100

# En fonction du epsilon demandé, la bonne valeur de L

def goodL(epsilon):
    res = - np.log(epsilon) / np.log(2) + 1
    return int(res)


# En fonction de la valeur de L et l, la bonne valeur du nombre de simu

def nbDeSimu(L, l, e):
    return int((L / (e ** 2 * 2.0 ** l)))



#TODO compléter cette fonction de Nombre de pas dans une méthode multipas avec le schéma de multipas_milstein

def nbDeSimu_milstein(L, l, e):
    if (l < 3):
        return 5000000
    return int((L / (e ** 2 * 2.0 ** l)))


def trancheN(l, M):
    dt = 1.0 / (2 ** l)
    meanW = 0
    meanW2 = 0
    Y = np.ones(M) * S0
    X = np.ones(M) * S0
    for j in range(2 ** (l - 1)):
        dW = np.random.normal(0, 1, [2, M]) * np.sqrt(dt)
        Y = Y + sigma * Y * (dW[0] + dW[1]) + r  * Y *2.0* dt
        X = X + sigma * X * (dW[0]) + r * X * dt
        X = X + sigma * X * (dW[1]) + r * X * dt
        meanW += np.mean(dW)
        meanW2 += dW[0] ** 2 + dW[1] ** 2
    # Ici, on prend la valeur du call, "A la main", en quelque sorte
    X = callPayOff(X,K)
    Y = callPayOff(Y,K)
    Xmean = np.mean(X)
    Ymean = np.mean(Y)
    totalSum = Xmean - Ymean
    var = X-Y
    variance = var **2 - np.mean(var)**2
    variance  = np.mean(variance)
    meanW2 = meanW2 - meanW ** 2
    meanW2 = np.mean(meanW2)
    #assert meanW < 0.01
    #assert meanW > -0.01
    #assert meanW2 > 0.95
    #assert meanW2 < 1.05

    return totalSum, variance


def trancheN_milstein(l, M):
    dt = T / (2 ** l)
    meanW = 0
    meanW2 = 0
    Y = np.ones(M) * S0
    X = np.ones(M) * S0
    bound = int(2**(l-1))
    for j in range(bound):
        dW = np.random.normal(0, 1, [2, M]) * np.sqrt(dt)
        Y = Y + sigma * Y * (dW[0] + dW[1]) + 0.5 * sigma **2 * Y * ((dW[0] + dW[1])**2 - (2*dt)) + r  * Y *2.0* dt
        X = X + sigma * X * (dW[0]) + 0.5 * sigma**2 * X * (dW[0] **2 - dt) + r * X * dt
        X = X + sigma * X * (dW[1]) + 0.5 * sigma**2 * X * (dW[1] **2 - dt) + r * X * dt
        meanW += np.mean(dW)
        meanW2 += dW[0] ** 2 + dW[1] ** 2
    # Ici, on prend la valeur du call, "A la main", en quelque sorte
    X = callPayOff(X,K)
    Y = callPayOff(Y,K)
    Xmean = np.mean(X)
    Ymean = np.mean(Y)
    totalSum = Xmean - Ymean
    var = X-Y
    variance = var **2 - np.mean(var)**2
    variance  = np.mean(variance)
    meanW2 = meanW2 - meanW ** 2
    meanW2 = np.mean(meanW2)
    #assert meanW < 0.01
    #assert meanW > -0.01
    #assert meanW2 > 0.95
    #assert meanW2 < 1.05

    return totalSum, variance

def tranche0_milstein(M):
    dW = np.random.normal(0,1, M)
    Y = S0 * np.ones(M)
    Y = Y + sigma*Y*dW + 0.5*sigma **2 * Y * (dW**2 - T**2) + r * Y * T
    Y =callPayOff(Y, K)
    return np.mean(Y)

def tranche0(M):
    dW = np.random.normal(0,1, M)
    Y = S0 * np.ones(M)
    Y = Y + sigma*Y*dW + r*Y*T
    Y =callPayOff(Y, K)
    variance = Y **2 - np.mean(Y)**2
    variance  = np.mean(variance)
    return np.mean(Y) , variance

def callPayOff(X,K):
    n = X.shape
    Y = X-K* np.ones(n)
    Y = Y.clip(0)
    return Y


def multipas(L, epsilon):
    pas = np.zeros(L)
    M = nbDeSimu(L, 0, epsilon)
    pas[0] = tranche0(M)
    for l in range(1,L):
        M = nbDeSimu(L,l,epsilon)
        res, var =trancheN(l, M)
        pas[l] = res
    return np.sum(pas)

def multipas_milstein(L, epsilon):
    L = int(L)
    pas = np.zeros(L)
    M = nbDeSimu_milstein(L, 0, epsilon)
    pas[0] = tranche0_milstein(M)
    for l in range(1,L):
        M = nbDeSimu_milstein(L,l,epsilon)
        res, var =trancheN_milstein(l, M)
        pas[l] = res
    return np.sum(pas)
