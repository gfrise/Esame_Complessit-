#!/usr/bin/env python3
"""
simple_bootstrap_coverage_multi_extended.py

Versione semplice (stile tuo) che:
 - calcola coverage percentile-bootstrap e coverage normale
 - struttura i risultati separati per vari valori di alpha (Pareto) e df (t-student)
 - aggiunge Ornstein-Uhlenbeck e fractional Gaussian noise (fGn)
 - NON salva nulla su disco
 - mostra un unico grafico: coverage (più curve)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.linalg import toeplitz, cholesky

# ---------------------------
# Parametri (modifica se vuoi)
# ---------------------------
R = 300            # ripetizioni Monte Carlo per ogni n
B = 1000           # bootstrap replicates per campione
n_list = [20, 50, 100, 500, 2000]
alpha_ci = 0.05
z975 = norm.ppf(1 - alpha_ci/2)
rng = np.random.default_rng(12345)

# Parametri richiesti
pareto_alphas = [0.4, 1.0, 5.0, 20.0]
t_dfs = [0.4, 1.0, 5.0, 20.0]

# Parametri per OU e fGn (scelti come esempi; puoi modificarli)
ou_thetas = [0.01, 0.5, 2.0]    # velocità di ritorno (theta)
fgn_hurst = [0.5, 0.75, 0.9]    # H values

# ---------------------------
# Funzione bootstrap (stessa tua)
# ---------------------------
def bootstrap_means(x, B, rng):
    n = len(x)
    idx = rng.integers(0, n, size=(B, n))
    return x[idx].mean(axis=1)

# ---------------------------
# Sampler (stile tuo)
# ---------------------------
def sampler_normal(n, rng):
    return rng.normal(0.0, 1.0, size=n)

def sampler_t(n, rng, df):
    return rng.standard_t(df=df, size=n)

def sampler_pareto(n, rng, alpha_p):
    u = rng.random(n)
    return (1 - u)**(-1.0/alpha_p)

# Ornstein-Uhlenbeck discrete-time (simple Euler-like)
def sampler_ou(n, rng, theta=0.5, mu=0.0, x0=0.0):
    x = np.zeros(n)
    x[0] = x0
    for t in range(1, n):
        x[t] = x[t-1] + theta * (mu - x[t-1]) + rng.normal()
    return x

# fractional Gaussian noise via Cholesky (zero-mean)
def sampler_fgn(n, rng, hurst=0.75):
    # autocovariance for fractional Gaussian noise increments
    def gamma(k):
        return 0.5 * (abs(k+1)**(2*hurst) - 2*abs(k)**(2*hurst) + abs(k-1)**(2*hurst))
    cov = np.array([gamma(k) for k in range(n)])
    # toeplitz covariance matrix and cholesky
    L = cholesky(toeplitz(cov), lower=True)
    z = rng.normal(size=n)
    return L.dot(z)

# ---------------------------
# Helper: mean teorica quando definita
# ---------------------------
def pareto_pop_mean(alpha_p):
    return alpha_p / (alpha_p - 1.0) if alpha_p > 1.0 else None

def t_pop_mean(df):
    return 0.0 if df > 1.0 else None

# ---------------------------
# Core: valuta coverage e stampa (per un sampler dato)
# ---------------------------
def evaluate_for_sampler(label, sampler_fn, pop_mean):
    results = []
    print(f"\n--- {label} (pop_mean={'known' if pop_mean is not None else 'unknown'}) ---")
    header = f"{'n':>6}  {'cov_perc':>8}  {'cov_norm':>8}"
    print(header)
    print("-" * len(header))
    for n in n_list:
        cover_perc = 0
        cover_norm = 0
        for rep in range(R):
            x = sampler_fn(n, rng)
            m = float(np.mean(x))
            s = float(np.std(x, ddof=1))

            boot = bootstrap_means(x, B, rng)
            pl, ph = np.percentile(boot, [100*alpha_ci/2, 100*(1-alpha_ci/2)])

            norm_l = m - z975 * s / np.sqrt(n)
            norm_h = m + z975 * s / np.sqrt(n)

            if pop_mean is not None:
                cover_perc += (pl <= pop_mean <= ph)
                cover_norm += (norm_l <= pop_mean <= norm_h)

        cov_perc_rate = (cover_perc / R) if pop_mean is not None else np.nan
        cov_norm_rate = (cover_norm / R) if pop_mean is not None else np.nan

        print(f"{n:6d}  {cov_perc_rate:8.3f}  {cov_norm_rate:8.3f}")

        results.append({
            "label": label,
            "n": n,
            "pop_mean_known": pop_mean is not None,
            "coverage_percentile": cov_perc_rate,
            "coverage_normal": cov_norm_rate
        })
    return results

# ---------------------------
# Esecuzioni: normal baseline
# ---------------------------
all_results = []
all_results += evaluate_for_sampler("Normal(0,1)", sampler_normal, pop_mean=0.0)

# t-student con diversi df (stampa separata per ogni df)
for df in t_dfs:
    label = f"t(df={df})"
    pm = t_pop_mean(df)
    samp = lambda n, r, df_local=df: sampler_t(n, r, df_local)
    all_results += evaluate_for_sampler(label, samp, pop_mean=pm)

# Pareto con diversi alpha
for alpha_p in pareto_alphas:
    label = f"Pareto(alpha={alpha_p})"
    pm = pareto_pop_mean(alpha_p)
    samp = lambda n, r, a=alpha_p: sampler_pareto(n, r, a)
    all_results += evaluate_for_sampler(label, samp, pop_mean=pm)

# OU per diversi theta
for theta in ou_thetas:
    label = f"OrnsteinUhlenbeck(theta={theta})"
    samp = lambda n, r, th=theta: sampler_ou(n, r, theta=th, mu=0.0, x0=0.0)
    # OU has stationary mean mu=0
    all_results += evaluate_for_sampler(label, samp, pop_mean=0.0)

# fGn per diversi H
for H in fgn_hurst:
    label = f"fGn(H={H})"
    samp = lambda n, r, Hval=H: sampler_fgn(n, r, hurst=Hval)
    # fGn has mean 0 (constructed zero-mean)
    all_results += evaluate_for_sampler(label, samp, pop_mean=0.0)

# ---------------------------
# Plot: coverage (solo per casi con pop_mean_known True)
# ---------------------------
# raggruppa risultati per label
groups = {}
for r in all_results:
    groups.setdefault(r["label"], []).append(r)
for lbl in groups:
    groups[lbl] = sorted(groups[lbl], key=lambda x: x["n"])

plt.figure(figsize=(12,7))
for lbl, rows in groups.items():
    if any(r["pop_mean_known"] for r in rows):
        ns = [r["n"] for r in rows if r["pop_mean_known"]]
        covp = [r["coverage_percentile"] for r in rows if r["pop_mean_known"]]
        covn = [r["coverage_normal"] for r in rows if r["pop_mean_known"]]
        # plot percentile and normal as paired curves (small markers)
        plt.plot(ns, covp, marker='o', linestyle='-', label=f"{lbl} - perc")
        plt.plot(ns, covn, marker='x', linestyle='--', label=f"{lbl} - norm")

plt.xscale('log')
plt.axhline(1-alpha_ci, color='gray', linestyle=':', label=f"target {1-alpha_ci:.2f}")
plt.xlabel("n (log scale)")
plt.ylabel("empirical coverage")
plt.title("Coverage: percentile bootstrap vs normal approx (multiple params & processes)")
plt.legend(ncol=2, fontsize='small')
plt.grid(True, which='both', linestyle=':', linewidth=0.4)
plt.tight_layout()
plt.show()

print("\nFatto. Nota: fGn (Cholesky) e ripetizioni con n grandi possono essere costose in termini di tempo/memoria.")
