import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import toeplitz, cholesky

# Parameters and Bootstrap
seed = 42
rng = np.random.default_rng(seed)

def bootstrap_means(x, B):
    idx = rng.integers(0, len(x), size=(B, len(x)))
    return x[idx].mean(axis=1)


# Distributions 
def gen_pareto(a, n):
    return (1 - rng.random(n))**(-1 / a)

def gen_t(df, n):
    return rng.standard_t(df, size=n)

def gen_ou(gamma, n):
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t-1] + gamma * (0.0 - x[t-1]) + rng.normal()
    return x

def gen_fgn(H, rng, n):
    def gamma(k):
        return 0.5 * (abs(k+1)**(2*H) - 2*abs(k)**(2*H) + abs(k-1)**(2*H))
    cov = np.array([gamma(k) for k in range(n)])
    L = cholesky(toeplitz(cov), lower=True)
    return L @ rng.normal(size=n)


# Repeated experiments
def run_experiment(params, generator, n, B, R):
    skew_mean = []
    skew_std = []
    kurt_mean = []
    kurt_std = []

    for p in params:
        skews = []
        kurts = []
        for r in range(R):
            x = generator(p, n)        
            m = bootstrap_means(x, B)
            skews.append(stats.skew(m))
            kurts.append(stats.kurtosis(m))   # excess kurtosis

        skew_mean.append(np.mean(skews))
        skew_std.append(np.std(skews, ddof=1))
        kurt_mean.append(np.mean(kurts))
        kurt_std.append(np.std(kurts, ddof=1))

    return (np.array(skew_mean), np.array(skew_std),
            np.array(kurt_mean), np.array(kurt_std))


# Plot
def plot_shape(param, sm, ss, km, ks, xlabel, title):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].errorbar(param, sm, yerr=ss, fmt='o-', markersize=4, lw=1.5,
                    ecolor='gray', elinewidth=1, capsize=3, color='#1f77b4')
    axs[0].axhline(0, ls='--', lw=1)
    axs[0].set_xlabel(xlabel); axs[0].set_ylabel('Skewness')
    axs[0].grid(alpha=0.3)

    axs[1].errorbar(param, km, yerr=ks, fmt='o-', markersize=4, lw=1.5,
                    ecolor='gray', elinewidth=1, capsize=3, color='#ff7f0e')
    axs[1].axhline(0, ls='--', lw=1)
    axs[1].set_xlabel(xlabel); axs[1].set_ylabel('Excess kurtosis')
    axs[1].grid(alpha=0.3)

    fig.suptitle(title)
    plt.show()


#Execution
n, B, R = 3000, 3000, 100

alphas = np.linspace(0.3, 10.0, 120)
s_mean, s_std, k_mean, k_std = run_experiment(alphas, gen_pareto, n, B, R)
plot_shape(alphas, s_mean, s_std, k_mean, k_std, 'alpha', 'Bootstrap mean shape — Pareto')

dfs = np.linspace(0.5, 30.0, 120)
s_mean, s_std, k_mean, k_std = run_experiment(dfs, gen_t, n, B, R)
plot_shape(dfs, s_mean, s_std, k_mean, k_std, 'df', 'Bootstrap mean shape — Student-t')

thetas = np.linspace(0.01, 5.0, 120)
s_mean, s_std, k_mean, k_std = run_experiment(thetas, gen_ou, n, B, R)
plot_shape(thetas, s_mean, s_std, k_mean, k_std, 'theta', 'Bootstrap mean shape — OU process')

Hs = np.concatenate([np.linspace(0.05, 0.5, 40), np.linspace(0.5, 0.95, 40)])
s_mean, s_std, k_mean, k_std = run_experiment(Hs, gen_fgn, n, B, R)
plot_shape(Hs, s_mean, s_std, k_mean, k_std, 'H', 'Bootstrap mean shape — fractional Gaussian noise')
