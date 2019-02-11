#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Le but de ce script est de répliquer les simulations du papier :
Alexander Goldenshluger. A universal procedure for aggregating estimators. The Annals
of Statistics, pages 542{568, 2009.
"""

# Imports.
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random as rd
import math
from scipy.optimize import minimize

A = np.array([[2, 4], [5, -6]])
B = np.array([[9, -3], [3, 6]])


# Initialisation des paramètres.
n = 1000
sigma = np.identity(n)
p = 2
nb_experiments = 100

for K in [5, 50, 250, 500]:

    rank = []
    index = []
    mean_risk =[]

    for experiment in range(nb_experiments):
        # Tirage de mu.
        mu = np.zeros(n) # Initialise mu.
        indices = rd.sample(range(n), K) # Choisit les indices à remplir.
        for i in indices:
            mu[i] = np.random.normal(0, 1) # Remplit les indices avec des variables gaussiennes.


        # Tirage de Y1, Y2.
        epsilon_1 = 0.5
        epsilon_2 = 1
        Y1 = np.random.multivariate_normal(mu, (epsilon_1 ** 2) * sigma)
        Y2 = np.random.multivariate_normal(mu, (epsilon_2 ** 2) * sigma)


        # Calcul des 20 estimateurs.
        est = []
        od = [5, 10, 20, 50, 100, 200, 300, 500, 700, 800]
        for v in od:
            est.append(np.array([Y1[i] if i <= v else 0 for i in range(len(Y1)) ]))

        t = [1, n ** (1/4), n ** (1/2), n ** (3/4), n ** (5/6), n ** (7/8), n ** (9/10), n ** (15/16), n ** (31/32), n ** (63/64)]
        for v in t:
            thres = epsilon_1 * math.sqrt(2 * math.log(n / v))
            est.append(np.array([y if abs(y) >= thres else 0 for y in Y1 ]))


        # Calcul des vecteurs-tests Psi.
        Psi = []
        N = len(est)
        dist = lambda x, y : (abs(x - y) ** p) * np.sign(x - y)
        for i in range(N):
            for j in range(N):
                if j != i :
                    norme = np.linalg.norm(est[i] - est[j], p)
                    Psi.append(np.array([(dist(est[i][k], est[j][k]) / (norme) ** (p - 1)) for k in range(len(est[i])) ]))


        # Sélection de l'estimateur.
        M = []
        for i in range(N):
            M.append(max([ (abs((np.transpose(psi)).dot(Y2 - est[i]))) / (np.linalg.norm(psi, p)) for psi in Psi]))
        i_tilde = np.argmin(M)
        est_tilde = est[i_tilde]


        # Compare l'estimateur obtenu avec le meilleur estimateur.
        risks = [np.linalg.norm(est[i] - mu, p) for i in range(N)]
        optimal_risk = min(risks)
        experimental_risk = np.linalg.norm(est_tilde - mu, p)

        rank.append(1 + np.array(risks).argsort().argsort()[i_tilde]) # Rang de l'estimateur.
        index.append(i_tilde)
        mean_risk.append(np.array(risks))


    # Traite les données :
    hist_rank, b = np.histogram(rank, bins = 20, range = (0, 20))
    hist_index, b = np.histogram(index, bins = 20, range = (0, 20))
    hist_mean_risk = np.mean(mean_risk, axis=0)


    # Affiche et enregistre les données.
    data = [hist_rank, hist_index, hist_mean_risk]
    xaxes = ['Rank','Index','Index']
    yaxes = ['Selections','Selections','Risk']
    titles = ['','K = ' + str(K),'']

    f,a = plt.subplots(1, 3)
    a = a.ravel()
    for idx,ax in enumerate(a):
        ax.bar(range(20), data[idx])
        ax.set_title(titles[idx])
        ax.set_xlabel(xaxes[idx])
        ax.set_ylabel(yaxes[idx])
    plt.tight_layout()
    f.savefig('K=' + str(K) + '.png')   # Enregistre l'image.
    plt.close(f)
