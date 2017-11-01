
import numpy as np
import matplotlib.pyplot as plt
import os

#Premier test, simulationtion de deux mouvements browniens non correlés

#####

def clark_cameron(n,M):
    X1chapeau = np.zeros(M).astype('float64')
    X2chapeau = np.zeros(M).astype('float64')
    X1barre = np.zeros(M).astype('float64')
    X2barre = np.zeros(M).astype('float64')
    X1 = np.zeros(M).astype('float64')
    X2 = np.zeros(M).astype('float64')

    dt = 2**(-n)

    for i in range(int(2**(n-1))):

        dW = np.random.normal(0,1,[2,2,M]).astype('float64')
        dW = dW * np.sqrt(dt)

        #First coordinate is time, second is W2
        X2chapeau += X1chapeau * dW[0][1] + 0.5 * (dW[0][0]* dW[0][1])
        X1chapeau += dW[0][0]

        X2chapeau += X1chapeau * dW[1][1] + 0.5 * (dW[1][0]* dW[1][1])
        X1chapeau += dW[1][0]

        X2barre += X1barre * dW[1][1] + 0.5 * (dW[1][0]* dW[1][1])
        X1barre += dW[1][0]

        X2barre += X1barre * dW[0][1] + 0.5 * (dW[0][0]* dW[0][1])
        X1barre += dW[0][0]

        X2 += X1*(dW[0][1] + dW[1][1]) + 0.5*(dW[0][0]+ dW[1][0])*(dW[0][1] + dW[1][1])
        X1 += dW[0][0]+ dW[1][0]

    P1 = X2chapeau.clip(0)
    P1Var = np.var(P1, dtype = np.float64)
    P1mean = abs(np.mean(P1, dtype = np.float64))

    P2 = X2chapeau.clip(0) - X2.clip(0)
    P2Var = np.var(P2, dtype = np.float64)
    P2mean = abs(np.mean(P2, dtype = np.float64))

#    P3 = 0.5 * (np.cos(X2barre) + np.cos(X2chapeau)) - np.cos(X2)
    P3 = 0.5 * (X2barre.clip(0) + X2chapeau.clip(0)) - X2.clip(0)
    P3Var = np.var(P3, dtype=np.float64)
    P3mean = abs(np.mean(P3, dtype = np.float64))

    return [P1Var, P2Var, P3Var],[P1mean,P2mean, P3mean]
#
# N = 14
# M = 100000
#
# resultVar1 = np.zeros(N)
# resultVar2 = np.zeros(N)
# resultVar3 = np.zeros(N)
#
# resultMean1 = np.zeros(N)#Clark–Cameron SDEs
# resultMean2 = np.zeros(N)
# resultMean3 = np.zeros(N)
#
# fichier_res = open('PayOffClarksCameron.txt','w')
#
# fichier_res.write("=" * 40)
#
# fichier_res.write("Début de la séquence de tirages\n")
# for i in range(1, N+1):
# #    fichier_res.write("")
# #    fichier_res.write("*** Simulation avec %d pas " %(2**i) + "***")
#     print("***" , "Simulation avec %d pas" %(2**i), "***")
# #    fichier_res.write(str(M) + " simulations")
#     X,Y =clark_cameron(i, M)
#     resultVar1[i - 1] = np.log2(X[0])
#     resultVar2[i-1] = np.log2(X[1])
#     resultVar3[i-1] = np.log2(X[2])
#
#     resultMean1[i-1] = np.log2(Y[0])
#     resultMean2[i-1] = np.log2(Y[1])
#     resultMean3[i-1] = np.log2(Y[2])
#
#     fichier_res.write("var" + "\n")
#     fichier_res.write(str(resultVar1[i-1]) + ",")
#     fichier_res.write(str(resultVar2[i-1]) + ",")
#     fichier_res.write(str(resultVar3[i-1]) + "\n")
#     fichier_res.write("means"+ "\n")
#     fichier_res.write(str(resultMean1[i-1]) + ",")
#     fichier_res.write(str(resultMean2[i-1]) + ",")
#     fichier_res.write(str(resultMean3[i-1]) + "\n")
#
# fichier_res.close()
# #print(results)
#
#
#
# fig = plt.figure()
#
# plt.plot(resultVar1, linewidth=0.8, marker="*", label='$P_{l}$')
# plt.plot(resultVar2,linewidth=0.8, marker="+", label='$P_{l} - P_{l-1}$')
# plt.plot(resultVar3, linewidth=0.8, marker="x", label='$P^{an}_{l}- P_{l-1}$')
# plt.xlabel('level l')
# plt.ylabel('$log_{2}(variance)$')
#
#
# plt.legend(loc='lower left')
# plt.show()
#
#
#
# fig2 = plt.figure()
# plt.plot(resultMean1, linewidth=0.8, marker="*", label='$P_{l}$')
# plt.plot(resultMean2,linewidth=0.8, marker="+", label='$P_{l} - P_{l-1}$')
# plt.plot(resultMean3, linewidth=0.8, marker="x", label='$P^{an}_{l}- P_{l-1}$')
# plt.xlabel('level l')
# plt.ylabel('log2 |mean|')
# plt.legend(loc='best')
#
# plt.show()
#
# #fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
#
# #ax = fig.add_subplot(111)
# #fig.subplots_adjust(top=0.85)
# #ax.set_title('axes title')
#
#
# #ax.text(3, 8, 'boxed italics text in data coords', style='italic', bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
#
# #ax.text(2, 6, r'an equation: $E=mc^2$', fontsize=15)
#
# #ax.text(3, 2, u'unicode: Institut f\374r Festk\366rperphysik')
#
# #ax.text(0.95, 0.01, 'colored text in axes coords', verticalalignment='bottom', horizontalalignment='right',
# #        transform=ax.transAxes, color='green', fontsize=15)
#
#
# #ax.plot([2], [1], 'o')
# #ax.annotate('annotate', xy=(2, 1), xytext=(3, 4),
# #            arrowprops=dict(facecolor='black', shrink=0.05))
#
# #ax.axis([0, 10, 0, 10])
#
# #plt.show()
#
#
# os.system('gedit PayOffClarksCameron.txt')
