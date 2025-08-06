import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd

import numpy as np, matplotlib.pyplot as plt
from scipy.stats import t,pareto

N, i, M, SD, df, alpha = 10**3, 1000, 3, 0.4, 5, 1.5

def OU(N,M,SD,theta,dt=1):
    x = np.empty(N)
    x[0] = M
    for t in range(1, N):
        x[t] = x[t-1] + theta*(M - x[t-1])*dt + SD*np.sqrt(dt)*np.random.randn()
    return x

# means_t = [np.mean(t.rvs(df=df, loc=M, scale=SD/np.sqrt(df/(df-2)), size=N)) for _ in range(i)]
# mean_pl = [np.mean(pareto.rvs(alpha, size=N)) for _ in range(i)]
means_gauss = [np.mean(np.random.normal(M,SD,N)) for _ in range(i)]
means_ou_low = [np.mean(OU(N,M,SD,theta=1.5)) for _ in range(i)]   #θ grande poca memoria
means_ou_high= [np.mean(OU(N,M,SD,theta=0.001)) for _ in range(i)]  #θ piccolo molta memoria

def plotta(data, title, M, SD, N):
    emp_mean = np.mean(data)
    emp_std = np.std(data)*np.sqrt(N)

    plt.hist(data, bins=30, edgecolor="black")
    plt.title(
        f"{title}\n"
        f"μ bootstrap = {emp_mean:.3f}   σ bootstrap = {emp_std:.3f}\n"
        f"μ teorica = {M:.3f}          σ teorica = {SD:.3f}"
    )    
    plt.xlabel("Media campionaria")
    plt.ylabel("Frequenza")
    plt.show()

plotta(means_gauss, "iid Gaussiana", M, SD, N)
plotta(means_ou_low, "OU memoria breve", M, SD, N)
plotta(means_ou_high, "OU memoria lunga", M, SD, N)

# Impostazioni generali
np.random.seed(42)
sample_size = 1000
n_bootstrap = 1000

def bootstrap_mean(sample, n_bootstrap=1000):
    """Esegue bootstrap con rimpiazzo per stimare la media."""
    means = np.array([
        np.mean(np.random.choice(sample, size=len(sample), replace=True))
        for _ in range(n_bootstrap)
    ])
    return means

def analyze_distribution(dist_name, sample, theoretical_mean, theoretical_std):
    """Analizza la distribuzione e stampa istogrammi e metriche."""
    boot_means = bootstrap_mean(sample, n_bootstrap)
    
    # Metriche bootstrap
    boot_mean = np.mean(boot_means)
    boot_std = np.std(boot_means)

    # Grafici
    plt.figure()
    plt.hist(sample, bins=50, alpha=0.6, density=True)
    plt.title(f"{dist_name} - Campione")
    plt.xlabel("Valore")
    plt.ylabel("Frequenza")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.hist(boot_means, bins=50, alpha=0.7, density=True)
    plt.title(f"{dist_name} - Bootstrap delle medie")
    plt.xlabel("Media")
    plt.ylabel("Frequenza")
    plt.grid(True)
    plt.show()

    # Risultati
    results = {
        "Distribuzione": dist_name,
        "Media teorica": theoretical_mean,
        "Media bootstrap": boot_mean,
        "Errore media": abs(boot_mean - theoretical_mean),
        "Std teorica": theoretical_std,
        "Std bootstrap delle medie": boot_std,
        "Errore std": abs(boot_std - theoretical_std)
    }

    return results

# Lista dei risultati
results = []

# 1. Normale standard
sample = np.random.normal(0, 1, sample_size)
results.append(analyze_distribution("Normale standard", sample, 0, 1 / np.sqrt(sample_size)))

# 2. Student t (df=1.5) - varianza fortemente divergente
df = 1.5
sample = stats.t.rvs(df, size=sample_size)
results.append(analyze_distribution("Student t (df=1.5)", sample, 0, np.nan))  # std teorica non definita

# 3. Student t (df=2.1) - varianza ancora divergente ma meno
df = 2.1
sample = stats.t.rvs(df, size=sample_size)
theo_std = np.sqrt(df / (df - 2)) / np.sqrt(sample_size)
results.append(analyze_distribution("Student t (df=2.1)", sample, 0, theo_std))

# 4. Student t (df=5) - varianza finita
df = 5
sample = stats.t.rvs(df, size=sample_size)
theo_std = np.sqrt(df / (df - 2)) / np.sqrt(sample_size)
results.append(analyze_distribution("Student t (df=5)", sample, 0, theo_std))

# 5. Pareto (α=1.5) - varianza molto divergente
alpha = 1.5
pareto_sample = (np.random.pareto(alpha, sample_size) + 1)
theo_mean = alpha / (alpha - 1) if alpha > 1 else np.inf
results.append(analyze_distribution("Pareto (α=1.5)", pareto_sample, theo_mean, np.nan))

# 6. Pareto (α=2.1) - varianza divergente ma meno
alpha = 2.1
pareto_sample = (np.random.pareto(alpha, sample_size) + 1)
theo_mean = alpha / (alpha - 1)
theo_std = np.sqrt(alpha / ((alpha - 1)**2 * (alpha - 2))) / np.sqrt(sample_size)
results.append(analyze_distribution("Pareto (α=2.1)", pareto_sample, theo_mean, theo_std))

# 7. Pareto (α=3.5) - varianza finita
alpha = 3.5
pareto_sample = (np.random.pareto(alpha, sample_size) + 1)
theo_mean = alpha / (alpha - 1)
theo_std = np.sqrt(alpha / ((alpha - 1)**2 * (alpha - 2))) / np.sqrt(sample_size)
results.append(analyze_distribution("Pareto (α=3.5)", pareto_sample, theo_mean, theo_std))

# Mostra tabella risultati
df_results = pd.DataFrame(results)
import caas_jupyter_tools as tools; tools.display_dataframe_to_user(name="Risultati Bootstrap", dataframe=df_results)

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from arch.bootstrap import MovingBlockBootstrap
import nolds
from fbm import FBM
from fracdiff import fracdiff
import warnings
warnings.filterwarnings("ignore")

# Parametri
n = 1000
n_bootstrap = 1000

def bootstrap_mean_classic(series, n_bootstrap=1000):
    return np.array([
        np.mean(np.random.choice(series, size=len(series), replace=True))
        for _ in range(n_bootstrap)
    ])

def bootstrap_mean_moving_block(series, block_size, n_bootstrap=1000):
    mbb = MovingBlockBootstrap(block_size, series)
    return np.array([np.mean(data[0]) for data in mbb.bootstrap(n_bootstrap)])

def plot_bootstrap_results(name, series, classic_means, block_means):
    plt.figure()
    plt.hist(series, bins=50, alpha=0.5, label="Serie originale", density=True)
    plt.title(f"{name} - Serie originale")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.hist(classic_means, bins=50, alpha=0.6, label="Bootstrap classico", density=True)
    plt.hist(block_means, bins=50, alpha=0.6, label="Block bootstrap", density=True)
    plt.title(f"{name} - Medie bootstrap")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"\n{name}")
    print("-" * 40)
    print(f"Media reale: {np.mean(series):.4f}")
    print(f"Std reale (della media): {np.std(series)/np.sqrt(len(series)):.4f}")
    print(f"Hurst stimato: {nolds.hurst_rs(series):.4f}")
    print(f"Bootstrap classico - Media: {np.mean(classic_means):.4f}, Std: {np.std(classic_means):.4f}")
    print(f"Block bootstrap - Media: {np.mean(block_means):.4f}, Std: {np.std(block_means):.4f}")

# 1. AR(1) con phi=0.8
phi = 0.8
ar = np.array([1, -phi])
ma = np.array([1])
ar_process = ArmaProcess(ar, ma)
series_ar1 = ar_process.generate_sample(nsample=n)

classic_ar1 = bootstrap_mean_classic(series_ar1, n_bootstrap)
block_ar1 = bootstrap_mean_moving_block(series_ar1, block_size=20, n_bootstrap=n_bootstrap)
plot_bootstrap_results("AR(1)", series_ar1, classic_ar1, block_ar1)

# 2. Fractional Brownian Motion con H=0.8
def generate_fbm(n, hurst):
    f = FBM(n=n, hurst=hurst, length=1, method='daviesharte')
    return f.fbm()

fbm_series = generate_fbm(n, hurst=0.8)
classic_fbm = bootstrap_mean_classic(fbm_series, n_bootstrap)
block_fbm = bootstrap_mean_moving_block(fbm_series, block_size=20, n_bootstrap=n_bootstrap)
plot_bootstrap_results("fBM (H=0.8)", fbm_series, classic_fbm, block_fbm)

# 3. ARFIMA(0,d,0) con d=0.4
def generate_arfima(d, n):
    white_noise = np.random.normal(size=n + 100)
    series, _ = fracdiff(white_noise, d)
    return series[-n:]

series_arfima = generate_arfima(d=0.4, n=n)
classic_arfima = bootstrap_mean_classic(series_arfima, n_bootstrap)
block_arfima = bootstrap_mean_moving_block(series_arfima, block_size=20, n_bootstrap=n_bootstrap)
plot_bootstrap_results("ARFIMA (d=0.4)", series_arfima, classic_arfima, block_arfima)

print("\n✔️ Analisi completata.")
