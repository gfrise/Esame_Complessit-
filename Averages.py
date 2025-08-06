import numpy as np 

def time_average(x, tau):
    """
    Calcola la autocovarianza time average:
    (1/T) * sum_t x[t] * x[t+tau]
    
    x : lista o array monodimensionale (una sola serie)
    tau : lag (intero >= 0)
    """
    T = len(x)
    somma = 0.0
    for t in range(T - tau):
        somma += x[t] * x[t + tau]
    return somma / T

# def time_avg(x,tau):
#     n = len(x)-tau
#     acc = 0
#     for i in range(n):
#         acc+=x[i]*x[i+tau]
#     return acc/n
# for _ in range(m):
#     x = np.empty(nn)
#     x[0] = 0.3
#     noise = np.random.normal(0,3,nn)
#     for i in range(1,nn):
#         x[i]=x[i-1]-y*dt*x[i-1]+np.sqrt(dt)*noise[i]
#         ac[i]+=x[0]*x[k]
# Array per memorizzare traiettorie e autocorrelazioni
# all_trajectories = np.zeros((nENS, nR))
# ensemble_means = np.zeros(nENS)
# autocorrs = np.zeros((nENS, tau_max))

# for iter in range(nENS):
#     # Generazione rumore gaussiano
#     total_steps = nR * enne
#     Z = np.random.normal(0, np.sqrt(2 * delt), total_steps)
    
#     # Simulazione processo Ornstein-Uhlenbeck
#     X = np.zeros(total_steps)
#     X[0] = np.random.uniform(-1, 1)
    
#     for i in range(1, total_steps):
#         X[i] = X[i-1] - gammaOU * X[i-1] * delt + Z[i]
    
#     # Campionamento
#     sampled_X = X[::enne]

#     all_trajectories[iter] = sampled_X
#     ensemble_means[iter] = np.mean(sampled_X)
    
#     # Calcolo autocorrelazione per questa traiettoria
#     autocorrs[iter] = autocorrelation(sampled_X, tau_max)

# # Calcolo autocorrelazione media sull'ensemble
# avg_autocorr = np.mean(autocorrs, axis=0)

def mixed_average(series, tau):
    """
    Calcola la autocovarianza mixed average:
      C(τ) = (1/M) * sum_{i=1}^M [ (1/T) * sum_{t=0}^{T-τ-1} series[i][t] * series[i][t+τ] ]

    series : lista di M liste, ciascuna di lunghezza T
    tau    : lag (intero >= 0)
    """
    M = len(series)          # numero di serie
    T = len(series[0])       # lunghezza di ciascuna serie
    somma_totale = 0.0

    for seq in series:
        somma_i = 0.0
        for t in range(T - tau):
            somma_i += seq[t] * seq[t + tau]
        somma_totale += somma_i / T

    return somma_totale / M

def ensemble_average(X, tau):
    """
    Calcola la autocovarianza ensemble average:
    (1/M) * sum_i x_i[0] * x_i[tau]
    
    X : lista di liste o 2D array-like, con shape (M, T)
    tau : lag (intero >= 0)
    """
    M = len(X)          # numero di serie
    somma = 0.0

    for i in range(M):
        somma += X[i][0] * X[i][tau]

    return somma / M
