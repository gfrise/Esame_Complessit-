import numpy as np 

def time_average(x, tau):
    """
    (1/T) * sum_t x[t] * x[t+tau]
    """
    T = len(x)
    somma = 0.0
    for t in range(T - tau):
        somma += x[t] * x[t + tau]
    return somma / T

# for _ in range(m):
#     x = np.empty(nn)
#     x[0] = 0.3
#     noise = np.random.normal(0,3,nn)
#     for i in range(1,nn):
#         x[i]=x[i-1]-y*dt*x[i-1]+np.sqrt(dt)*noise[i]
#         ac[i]+=x[0]*x[k]
# Array per memorizzare traiettorie e autocorrelazioni
autocorrs = np.zeros((nENS, tau_max))
for i in range(m):
    # Calcolo autocorrelazione per questa traiettoria
    autocorrs[i] = ac(OU(),taumax)
# Calcolo autocorrelazione media sull'ensemble
avg_autocorr = np.mean(autocorrs,axis=0)


def ensemble():
    mean_ens, mean2_ens = np.zeros(N),np.zeros(N)
    for _ in range(m):
        traj = OU(1)
        mean_ens[:]+=traj
        mean2_ens[:]+=traj**2
    mean_ens[:]/=m
    mean2_ens[:]/=m

def mixed_average(series, tau):
    """
    Calcola la autocovarianza mixed average:
      C(τ) = (1/M) * sum_{i=1}^M [ (1/T) * sum_{t=0}^{T-τ-1} series[i][t] * series[i][t+τ] ]

    series : lista di M liste, ciascuna di lunghezza T
    """
    M = len(series)          # numero di serie
    T = len(series[0])       # lunghezza serie
    somma_totale = 0.0

    for seq in series:
        somma_i = 0.0
        for t in range(T - tau):
            somma_i += seq[t] * seq[t + tau]
        somma_totale += somma_i / T

    return somma_totale / M

def ensemble_average(X, tau):
    """
    (1/M) * sum_i x_i[0] * x_i[tau]
    
    X : lista di liste o 2D array-like, con shape (M, T)
    """
    M = len(X)          # numero di serie
    somma = 0.0

    for i in range(M):
        somma += X[i][0] * X[i][tau]

    return somma / M

# Compito Wiener:
#   - Moto Browniano (processo di Wiener)
#   - PDF e varianza in funzione del tempo
Z_w = np.random.normal(0, np.sqrt(dt), N)
X_wiener = np.zeros(N)
X_wiener[0] = 0.1
for i in range(1, N):
    X_wiener[i] = X_wiener[i-1] + Z_w[i]
####