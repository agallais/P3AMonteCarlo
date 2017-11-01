
import numpy as np
import matplotlib.pyplot as plt
import math

N=10 #nombre de pas
T=1.
r=0.05
sigma=0.3
Y=100
M=5 #nombre simulations


#simulation des W
print ('simulation des W...')
aleatoire=np.sqrt(T/N)*np.random.randn(M,N)
aleatoire=np.concatenate((np.zeros(M).reshape(M,1),aleatoire), axis=1)
W=np.cumsum(aleatoire,axis=1)
print( 'done')

#solution exacte
print ('solution exacte...')
X=np.zeros((M,N+1))
X[:,0]=Y
for i in range(1,N+1):
    X[:,i]=Y*np.exp(sigma*W[:,i]+(r-(sigma**2)/2)*i*T/N)
print ('done')

#Discretisation (modele de BS)
def Xbarre(k):
    Xbarre=np.zeros((M,k))
    Xbarre[:,0]=Y
    for i in range(k-1):
        Xbarre[:,i+1]=Xbarre[:,i]+Xbarre[:,i]*sigma*(W[:,(i+1)*N/k]-W[:,i*N/k])+Xbarre[:,i]*r*T/(k-1)
    return Xbarre

#definition de f:
def f(K,X):
    m=X.shape[1]
    for i in range(0,M):
        if X[i,m-1]-K>0:
            X[i,m-1]=X[i,m-1]-K
        else:
            X[i,m-1]=0
    S=np.sum(X[:,m-1])/M
    return S

#esperance difference
def diff(X,Xbarre,K):
    return f(K,Xbarre)-f(K,X)

#convergence faible
def CVfaible(N,X,K):
    resultats=[]
    ran=[100,1000,2500,5000,10000, 20000,30000,40000, 50000,60000,70000,80000,90000]
    for i in ran:
        print ('iteration =',i)
        xbarre=Xbarre(i)
        x=X
        resultats.append(i*np.abs(diff(x,xbarre,K)))
    return ran,resultats

res=CVfaible(N, X, 100)


plt.figure("Convergence faible")
plt.scatter(np.log(res[0]),np.log(res[1]))
plt.show()
