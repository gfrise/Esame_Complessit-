import numpy as np
import sys
import matplotlib.pyplot as plt

def read_parameters(filename):
    with open(filename, 'r') as f:
        params = [float(line.strip()) for line in f.readlines()]
    return params

def autocorrelation(x, max_lag=None):
    """Calcola la funzione di autocorrelazione normalizzata"""
    n = len(x)
    if max_lag is None:
        max_lag = n // 2
    else:
        max_lag = min(max_lag, n // 2)
    
    mean = np.mean(x)
    var = np.var(x)
    acf = np.zeros(max_lag)
    
    for lag in range(max_lag):
        acf[lag] = np.mean((x[:n-lag] - mean) * (x[lag:] - mean)) / var
    
    return acf

# Lettura parametri
params = read_parameters("input_parametersOUmom.dat")
gammaOU = params[0]
enne = int(params[1])
delt = 1.0 / enne
nENS = int(params[2])
nR = int(sys.argv[1])
tau_max = nR // 2  # Lags massimi per autocorrelazione

print(f"Parametro gamma: {gammaOU}")
print(f"Time step: {delt}")
print(f"Dimensione ensemble: {nENS}")
print(f"Punti generati per traiettoria: {nR}")

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

# Salvataggio risultati autocorrelazione
with open("autocorrelation.dat", "w") as f:
    for lag, ac in enumerate(avg_autocorr):
        f.write(f"{lag} {ac}\n")

# Plot autocorrelazione (opzionale)
plt.figure(figsize=(10, 6))
plt.plot(range(tau_max), avg_autocorr, 'b-', linewidth=2)
plt.axhline(0, color='k', linestyle='--')
plt.xlabel('Lag (τ)', fontsize=14)
plt.ylabel('Autocorrelazione', fontsize=14)
plt.title(f'Autocorrelazione Processo OU (γ={gammaOU})', fontsize=16)
plt.grid(alpha=0.3)
plt.savefig('autocorrelation.png', dpi=150)
plt.close()

print("Calcolo autocorrelazione completato!")