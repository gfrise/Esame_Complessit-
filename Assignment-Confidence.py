import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz, cholesky

#Parameters
seed = 42
rng = np.random.default_rng(seed)

n, B, R = 300, 300, 100

alpha = 0.05

# Distributions
def gen_pareto(a, n):
    return (1 - rng.random(n)) ** (-1 / a)

def gen_student(df, n):
    return rng.standard_t(df, size=n)

def gen_ou(theta, n):
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t-1] - theta * x[t-1] + rng.normal()
    return x

def gen_fgn(H, n):
    def g(k):
        return 0.5 * (abs(k+1)**(2*H) - 2*abs(k)**(2*H) + abs(k-1)**(2*H))
    cov = np.array([g(k) for k in range(n)])
    L = cholesky(toeplitz(cov), lower=True)
    return L @ rng.normal(size=n)

# Means
mu_p = lambda a: np.nan if a <= 1 else a / (a - 1)
mu_t = lambda df: np.nan if df <= 1 else 0.0
mu_0 = lambda _: 0.0

#Bootstrap
def bootstrap_means(x, B):
    idx = rng.integers(0, len(x), size=(B, len(x)))
    return x[idx].mean(axis=1)

#Coverage and Percentiles
def coverage_and_percentiles(generator, mu_func, grid, n, B, R):
    cov = []
    se = []
    pct_list = []

    for p in grid:
        true = mu_func(p)
        if not np.isfinite(true):
            cov.append(np.nan)
            se.append(np.nan)
            pct_list.append(np.full(R, np.nan))
            continue

        hits = 0
        perc = np.empty(R)

        for r in range(R):
            x = generator(p, n)              
            m = bootstrap_means(x, B)        
            lo, hi = np.percentile(m, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            if lo <= true <= hi:
                hits += 1
            perc[r] = np.mean(m <= true)

        c = hits / R
        cov.append(c)
        se.append(np.sqrt(c * (1 - c) / R))
        pct_list.append(perc)

    return np.array(cov), np.array(se), np.vstack(pct_list)

# Cramer von mises, percentiles' reliability
def cramervonmises_w2(u):
    u = u[~np.isnan(u)]
    n_u = u.size
    if n_u == 0:
        return np.nan
    us = np.sort(u)
    i = np.arange(1, n_u + 1)
    expected = (2 * i - 1) / (2 * n_u)
    return 1 / (12 * n_u) + np.sum((us - expected) ** 2)

def affidabilita_from_percentiles(pct):
    w2 = np.array([cramervonmises_w2(row) for row in pct])
    finite = np.isfinite(w2)
    if finite.sum() == 0:
        return np.full_like(w2, np.nan, dtype=float)

    wmin, wmax = w2[finite].min(), w2[finite].max()
    if np.isclose(wmin, wmax):
        A = np.ones_like(w2, dtype=float)
    else:
        A = 1 - (w2 - wmin) / (wmax - wmin)

    A[~finite] = np.nan
    return A

# Plot
def plot_results(param, cov, se, pct, xlabel, title):
    A = affidabilita_from_percentiles(pct)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    color_cov = '#1f77b4'
    lower, upper = cov - se, cov + se

    axs[0].fill_between(param, lower, upper, where=np.isfinite(lower),
                        color=color_cov, alpha=0.08, linewidth=0, zorder=1)
    axs[0].errorbar(param, cov, yerr=se, fmt='none',
                    ecolor=color_cov, elinewidth=1.6, capsize=4, alpha=0.95, zorder=2, label='Confidence')
    axs[0].plot(param, cov, '-', lw=1.1, color=color_cov, zorder=3)
    axs[0].scatter(param, cov, s=40, facecolor='white', edgecolor=color_cov, linewidth=1.2, zorder=4)

    try:
        ref = 1 - alpha
    except NameError:
        ref = 0.95
    axs[0].axhline(ref, ls='--', color='gray', lw=1.2, alpha=0.9, zorder=5)
    axs[0].set_ylim(-0.02, 1.02) 
    axs[0].set_xlabel(xlabel)
    axs[0].grid(alpha=0.22, linestyle='--')
    axs[0].legend(edgecolor='lightgray', facecolor='white', framealpha=1)

    color_rel = '#d62728'
    axs[1].plot(param, A, 'o-', lw=1.6, markersize=5,
                color=color_rel, markerfacecolor='white', markeredgewidth=1.2, markeredgecolor=color_rel, zorder=3, label='Reliability')
    axs[1].set_ylim(-0.02, 1.02)
    axs[1].set_xlabel(xlabel)
    axs[1].axhspan(0.8, 1.02, color='#e6f5e6', alpha=0.45, zorder=0)
    axs[1].axhspan(0.6, 0.8,  color='#fff4cc', alpha=0.45, zorder=0)
    axs[1].axhspan(-0.02, 0.6, color='#fee0d2', alpha=0.45, zorder=0)

    for y in (0.6, 0.8, 0.90, 0.95):
        axs[1].axhline(y, ls='--', color='gray', lw=1.4, alpha=0.9, zorder=4)

    axs[1].grid(axis='y', alpha=0.22, linestyle='--')
    axs[1].legend(edgecolor='lightgray', facecolor='white', framealpha=1)

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.show()

# Free params of distributions
alphas = np.linspace(0.4, 20, 100)
dfs    = np.linspace(0.4, 20, 100)
thetas = np.linspace(0.01, 2, 100)
Hs     = np.linspace(0.5, 0.99999, 100)

# Execution
cov_p, se_p, pct_p = coverage_and_percentiles(gen_pareto, mu_p, alphas, n, B, R)
cov_t, se_t, pct_t = coverage_and_percentiles(gen_student, mu_t, dfs, n, B, R)
cov_o, se_o, pct_o = coverage_and_percentiles(gen_ou, mu_0, thetas, n, B, R)
cov_h, se_h, pct_h = coverage_and_percentiles(gen_fgn, mu_0, Hs, n, B, R)

plot_results(alphas, cov_p, se_p, pct_p, 'Alpha','Pareto (bootstrap)')
plot_results(dfs, cov_t, se_t, pct_t, 'Dof','Student-t (bootstrap)')
plot_results(thetas, cov_o, se_o, pct_o, 'Theta','OU process (bootstrap)')
plot_results(Hs, cov_h, se_h, pct_h, 'H','Fractional Gaussian Noise (bootstrap)')
