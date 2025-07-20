import numpy as np
import matplotlib.pyplot as plt

# Impostazioni
np.random.seed(42)
gamma, dt, T = 0.1, 0.1, int(1e5 / 0.1)
N, nbin = 100, 1000

# Simula processo OU
x = np.zeros(T)
for t in range(1, T):
    x[t] = x[t-1] - gamma * x[t-1] * dt + np.sqrt(dt) * np.random.normal(scale=np.sqrt(2))

# Calcola PDF bootstrap
bins = np.linspace(-5, 5, nbin+1)
pdfs = np.zeros((N, nbin))
for i in range(N):
    sample = x[np.random.randint(0, T, size=T)]
    pdfs[i], _ = np.histogram(sample, bins=bins, density=True)

# Media e deviazione standard delle PDF
centers = (bins[:-1] + bins[1:]) / 2
mean_pdf = pdfs.mean(axis=0)
std_pdf = pdfs.std(axis=0)

# Plot essenziale
plt.hist(x, bins=bins, density=True, alpha=0.5, label='PDF originale')
plt.errorbar(centers, mean_pdf, yerr=std_pdf, fmt='-', ecolor='r', capsize=2, label='Media ±1σ bootstrap')
plt.legend()
plt.tight_layout()
plt.show()
