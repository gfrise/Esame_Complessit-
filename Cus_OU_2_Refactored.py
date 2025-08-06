import numpy as np
import matplotlib.pyplot as plt

# Simulazione Ensemble-Average di:
#   • Ornstein–Uhlenbeck
#   • Processo multiscala di Risken
# Calcolo di pdf (media±σ), autocorrelazione (media±σ) e momenti centrali ordine 1…K.
# Tutti i parametri definiti internamente.

# def drift_OU(x, γ):        return -γ*x
# def drift_RISK(x, α, β):   return -α*x/(1+β*x*x)

t, step, taum, m, y = 300, 100, 50, 100, 0.1
dt, n, means, sd = 1/step, t*step, np.zeros(taum), np.zeros(taum)

def Ac(x):
    n = len(x) - taum
    m1, sd1, res = np.mean(x[:n]), np.std(x[:n]), np.zeros(taum)
    for t in range(taum):
        x2 = x[t:t+n]
        m2, sd2, corr = np.mean(x2), np.std(x2), np.mean(x[:n]*x2) #x[j]*x[j+t]
        res[t] = (corr - m1*m2) / (sd1 * sd2)
    return res

for k in range(m):
    x = np.zeros(n)
    x[0] = 0.1
    for i in range(1, n):
        x[i] = x[i-1] - y*x[i-1]*dt + np.random.normal(0,(2*dt)**0.5)
    ac = Ac(x[::step])
    means += ac
   # means[i] = np.mean(x[::step]) ?
    sd += ac**2

means /= m
sd = np.sqrt(sd/m-means**2)
plt.errorbar(np.arange(taum), means, yerr=sd, fmt='o', color='black')
plt.yscale("log")
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Ornstein-Uhlenbeck Autocorrelation')
plt.show()

