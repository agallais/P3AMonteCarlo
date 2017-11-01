'''
Created on 1 nov. 2016

@author: pierremusquar
'''
import numpy as np
import matplotlib.pyplot as plt

# import math

N = 2 ** 15# nombre de pas
T = 1.
r = 0.05
sigma = 0.3
Y = 1
M = 100  # nombre simulations

# simulation des W
print('simulation des W...')
aleatoire = np.sqrt(T / N) * np.random.randn(M, N)
aleatoire = np.concatenate((np.zeros(M).reshape(M, 1), aleatoire), axis=1)
W = np.cumsum(aleatoire, axis=1)
print('done')

# solution exacte
print('solution exacte...')
X = np.zeros((M, N))
X[:, 0] = Y
for i in range(1, N):
    X[:, i] = Y * np.exp(sigma * W[:, i] + (r - (sigma ** 2) / 2) * i * T / N)
print('done')


# solution exacte avec pas
def Xpas(X, k):
    x = np.zeros((M, 1))
    x[:, 0] = Y
    for j in range(1, k):
        x = np.concatenate((x, X[:, j * N / k].reshape(M, 1)), axis=1)
    return x


# Discretisation (modele de BS)
def Xbarre(k):
    Xbarre = np.zeros((M, k))
    Xbarre[:, 0] = Y
    for i in range(k - 1):
        Xbarre[:, i + 1] = Xbarre[:, i] + Xbarre[:, i] * sigma * (W[:, (i + 1) * N / k] - W[:, i * N / k]) + Xbarre[:,
                                                                                                             i] * r * T / (
                                                                                                             k - 1)
    return Xbarre


# esperance sup
def supp(p, X, Xbarre):
    u = np.abs(X - Xbarre) ** (2 * p)
    m = u.max(1)
    S = np.sum(m) / M
    return S


def CVforte(N, X, p):
    resultats = []
    ran = 1000 * np.arange(1, 30)
    for i in ran:
        print('nombre de pas =', i)
        xbarre = Xbarre(i)
        x = Xpas(X, i)
        resultats.append((i ** p) * supp(p, x, xbarre))
    return ran, resultats


res = CVforte(N, X, 1)
print(res)
plt.figure("Convergence forte")
plt.scatter(res[0], res[1], s=50)
plt.show()
