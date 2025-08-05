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
