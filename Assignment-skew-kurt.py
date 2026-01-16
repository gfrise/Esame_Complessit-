# bootstrap_shape_plots.py
"""
Plot skewness and kurtosis of bootstrap means vs distribution parameters
with multiple bootstrap repetitions to estimate variability and display
error bars. Improved aesthetics.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import toeplitz, cholesky

# ----------------------
# helpers
# ----------------------
def skew_kurtosis(x):
    return stats.skew(x), stats.kurtosis(x)


def bootstrap_means(x, B, rng):
    n = len(x)
    idx = rng.integers(0, n, size=(B, n))
    return x[idx].mean(axis=1)


# ----------------------
# generators
# ----------------------
def pareto_sample(alpha, size, rng):
    return (1 - rng.random(size)) ** (-1 / alpha)


def ornstein_uhlenbeck(n, theta, rng, mu=0.0, x0=0.0):
    x = np.zeros(n)
    x[0] = x0
    for t in range(1, n):
        x[t] = x[t-1] + theta * (mu - x[t-1]) + rng.normal()
    return x


def fgn(n, H, rng):
    gamma = lambda k: 0.5 * (
        abs(k + 1) ** (2 * H)
        - 2 * abs(k) ** (2 * H)
        + abs(k - 1) ** (2 * H)
    )
    cov = np.array([gamma(k) for k in range(n)])
    L = cholesky(toeplitz(cov), lower=True)
    return L @ rng.normal(size=n)


# ----------------------
# experiment runner with repeated bootstraps
# ----------------------
def run_experiment(param_values, generator, n=3000, B=3000, R=10, seed=0):
    rng = np.random.default_rng(seed)
    skews_mean = np.zeros(len(param_values))
    skews_std = np.zeros(len(param_values))
    kurts_mean = np.zeros(len(param_values))
    kurts_std = np.zeros(len(param_values))

    for i, p in enumerate(param_values):
        skews_r = []
        kurts_r = []
        for r in range(R):
            x = generator(p, rng)
            m = bootstrap_means(x, B, rng)
            s, k = skew_kurtosis(m)
            skews_r.append(s)
            kurts_r.append(k)
        skews_mean[i] = np.mean(skews_r)
        skews_std[i] = np.std(skews_r, ddof=1)
        kurts_mean[i] = np.mean(kurts_r)
        kurts_std[i] = np.std(kurts_r, ddof=1)

    return skews_mean, skews_std, kurts_mean, kurts_std


# ----------------------
# plotting with error bars and improved aesthetics
# ----------------------
def plot_shape(param, skew_mean, skew_std, kurt_mean, kurt_std, xlabel, title):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # Skewness
    ax[0].errorbar(param, skew_mean, yerr=skew_std, fmt='o-', markersize=4, lw=2,
                   ecolor='gray', elinewidth=1.2, capsize=3, color='#1f77b4')
    ax[0].axhline(0, ls='--', lw=1.5, alpha=0.5)
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel('Skewness')
    ax[0].grid(alpha=0.3)

    # Kurtosis
    ax[1].errorbar(param, kurt_mean, yerr=kurt_std, fmt='o-', markersize=4, lw=2,
                   ecolor='gray', elinewidth=1.2, capsize=3, color='#ff7f0e')
    ax[1].axhline(0, ls='--', lw=1.5, alpha=0.5)
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel('Excess kurtosis')
    ax[1].grid(alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.show()


# ----------------------
# main
# ----------------------
if __name__ == '__main__':

    n, B, R = 300, 300, 10

    # ----------------
    # Pareto alpha
    # ----------------
    alphas = np.linspace(0.3, 10.0, 120)
    s_mean, s_std, k_mean, k_std = run_experiment(
        alphas,
        lambda a, rng: pareto_sample(a, n, rng),
        n=n, B=B, R=R
    )
    plot_shape(alphas, s_mean, s_std, k_mean, k_std, 'alpha', 'Bootstrap mean shape — Pareto')

    # ----------------
    # Student-t df
    # ----------------
    dfs = np.linspace(0.5, 30.0, 120)
    s_mean, s_std, k_mean, k_std = run_experiment(
        dfs,
        lambda df, rng: rng.standard_t(df, size=n),
        n=n, B=B, R=R
    )
    plot_shape(dfs, s_mean, s_std, k_mean, k_std, 'df', 'Bootstrap mean shape — Student-t')

    # ----------------
    # OU theta
    # ----------------
    thetas = np.linspace(0.01, 5.0, 120)
    s_mean, s_std, k_mean, k_std = run_experiment(
        thetas,
        lambda th, rng: ornstein_uhlenbeck(n, th, rng),
        n=n, B=B, R=R
    )
    plot_shape(thetas, s_mean, s_std, k_mean, k_std, 'theta', 'Bootstrap mean shape — OU process')

    # ----------------
    # fGn H
    # ----------------
    Hs = np.concatenate([
        np.linspace(0.05, 0.5, 40),
        np.linspace(0.5, 0.95, 40),
    ])
    s_mean, s_std, k_mean, k_std = run_experiment(
        Hs,
        lambda H, rng: fgn(n, H, rng),
        n=n, B=B, R=R
    )
    plot_shape(Hs, s_mean, s_std, k_mean, k_std, 'H', 'Bootstrap mean shape — fractional Gaussian noise')
