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
