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

def gen_fgn(H, n):
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
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    axs[0].errorbar(param, sm, yerr=ss, fmt='o-', markersize=5, lw=2,
                    ecolor='lightgray', elinewidth=1.5, capsize=4, color='#1f77b4',
                    label='Skewness')
    axs[0].axhline(0, ls='--', lw=1.5, color='gray')
    axs[0].axhspan(-0.2, 0.2, color='green', alpha=0.1)  # banda riferimento
    axs[0].set_xlabel(xlabel, fontsize=12)
    axs[0].grid(alpha=0.25, linestyle='--')
    axs[0].legend()
    
    axs[1].errorbar(param, km, yerr=ks, fmt='s-', markersize=5, lw=2,
                    ecolor='lightgray', elinewidth=1.5, capsize=4, color='#d62728',
                    label='Excess Kurtosis')
    axs[1].axhline(0, ls='--', lw=1.5, color='gray')
    axs[1].axhspan(-0.2, 0.2, color='green', alpha=0.1)  # banda riferimento
    axs[1].set_xlabel(xlabel, fontsize=12)
    axs[1].grid(alpha=0.25, linestyle='--')
    axs[1].legend()

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.show()


#Execution
n, B, R = 300, 300, 100

alphas = np.linspace(0.4, 20.0, 100)
s_mean, s_std, k_mean, k_std = run_experiment(alphas, gen_pareto, n, B, R)
plot_shape(alphas, s_mean, s_std, k_mean, k_std, 'Alpha', 'Bootstrap mean shape — Pareto')

dfs = np.linspace(0.4, 20.0, 100)
s_mean, s_std, k_mean, k_std = run_experiment(dfs, gen_t, n, B, R)
plot_shape(dfs, s_mean, s_std, k_mean, k_std, 'Dof', 'Bootstrap mean shape — Student-t')

thetas = np.linspace(0.01, 2.0, 100)
s_mean, s_std, k_mean, k_std = run_experiment(thetas, gen_ou, n, B, R)
plot_shape(thetas, s_mean, s_std, k_mean, k_std, 'Theta', 'Bootstrap mean shape — OU process')

Hs = np.linspace(0.5, 0.99999, 100)
s_mean, s_std, k_mean, k_std = run_experiment(Hs, gen_fgn, n, B, R)
plot_shape(Hs, s_mean, s_std, k_mean, k_std, 'H', 'Bootstrap mean shape — Fractional Gaussian Noise')
