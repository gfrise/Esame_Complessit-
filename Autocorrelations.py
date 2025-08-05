import numpy as np, matplotlib.pyplot as plt
import random

def time_avg(x,tau):
    n = len(x)-tau
    acc = 0
    for i in range(n):
        acc+=x[i]*x[i+tau]
    return acc/n

def ensemble_avg

t = 100
dt = 1/t
n = 100
nn = t*n
y = 0.1
m = 50

for _ in range(m):
    x = np.empty(nn)
    x[0] = 0.3
    noise = np.random.normal(0,3,nn)
    for i in range(1,nn):
        x[i]=x[i-1]-y*dt*x[i-1]+np.sqrt(dt)*noise[i]
        ac[i]+=x[0]*x[k]
    
    
# Array per memorizzare traiettorie e autocorrelazioni
all_trajectories = np.zeros((nENS, nR))
ensemble_means = np.zeros(nENS)
autocorrs = np.zeros((nENS, tau_max))

for iter in range(nENS):
    # Generazione rumore gaussiano
    total_steps = nR * enne
    Z = np.random.normal(0, np.sqrt(2 * delt), total_steps)
    
    # Simulazione processo Ornstein-Uhlenbeck
    X = np.zeros(total_steps)
    X[0] = np.random.uniform(-1, 1)
    
    for i in range(1, total_steps):
        X[i] = X[i-1] - gammaOU * X[i-1] * delt + Z[i]
    
    # Campionamento
    sampled_X = X[::enne]

    all_trajectories[iter] = sampled_X
    ensemble_means[iter] = np.mean(sampled_X)
    
    # Calcolo autocorrelazione per questa traiettoria
    autocorrs[iter] = autocorrelation(sampled_X, tau_max)

# Calcolo autocorrelazione media sull'ensemble
avg_autocorr = np.mean(autocorrs, axis=0)