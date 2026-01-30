import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from scipy.linalg import toeplitz, cholesky

# Parameters and Boostratp
rng = np.random.default_rng(42)

n = 3000
B = 3000
bins = 60

def bootstrap_means(x):
    idx = rng.integers(0, len(x), (B, len(x)))
    return x[idx].mean(1)


# Print
def print_summary(label, x, m):
    sd = np.std(x, ddof=1)
    print(f"\n--- {label} ---")
    print(f"Mean (original)      {np.mean(x):.3f}")
    print(f"Mean (bootstrap)          {np.mean(m):.3f}")
    print(f"Std (original)       {sd:.3f}")
    print(f"Std (bootstrap)            {np.std(m,ddof=1):.3f}")
    print(f"Std (theoretical)      {sd/np.sqrt(len(x)):.3f}")
    print(f"2.5%                 {np.percentile(m,2.5):.3f}")
    print(f"97.5%                {np.percentile(m,97.5):.3f}")
    print(f"Skew                 {stats.skew(m):.3f}")
    print(f"Kurt (excess)        {stats.kurtosis(m):.3f}")


# plot
def plot_boot(title, m, x=None):
    plt.figure(figsize=(12,8))
    plt.hist(m, bins=bins, density=True, alpha=0.7, edgecolor='black')

    mu = np.mean(m)
    sd = np.std(m, ddof=1)
    xg = np.linspace(mu-6*sd, mu+6*sd, 300)
    plt.plot(xg, norm.pdf(xg, mu, 1.2*sd), 'r--', lw=2)

    p1, p2 = np.percentile(m, [2.5, 97.5])
    plt.axvline(p1, ls='--', lw=2, color='orange')
    plt.axvline(p2, ls='--', lw=2, color='orange')

    # stats
    rows = [
       ("Mean (bootstrap)", mu),
        ("Std (bootstrap)", sd),
        ("2.5%", p1),
        ("97.5%", p2),
        ("Skew", stats.skew(m)),
        ("Kurtosis (excess)", stats.kurtosis(m)),
    ]

    if x is not None:
        sx = np.std(x, ddof=1)
        rows = [
            ("Mean (original)", np.mean(x)),
            ("Std (original)", sx),
            ("Std (theoretical)", sx/np.sqrt(len(x))),
        ] + rows

    txt = "\n".join(f"{k:<19} {v:.4f}" for k,v in rows)

    plt.text(0.02, 0.98, txt,
             transform=plt.gca().transAxes,
             va='top', ha='left',
             fontsize=11, family='monospace',
             bbox=dict(boxstyle='round', fc='white', ec='black', alpha=0.9))


    plt.title(title)
    plt.xlabel("Bootstrap Means")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.show()


# EXPERIMENTS
def ornstein_uhlenbeck(theta, n):
    x = np.zeros(n)
    dt = 0.1
    for t in range(1, n):
        x[t] = x[t-1] + theta*(0.0 - x[t-1])*dt + rng.normal(0,np.sqrt(dt))
    return x

def fgn(hurst, n):
    def gamma(k):
        return 0.5 * (abs(k+1)**(2*hurst) - 2*abs(k)**(2*hurst) + abs(k-1)**(2*hurst))
    cov = np.array([gamma(k) for k in range(n)])
    L = cholesky(toeplitz(cov), lower=True)
    return L @ rng.normal(size=n)


# IID Uniform
x = rng.uniform(-2, 2, n)
m = bootstrap_means(x)
print_summary("IID Uniform", x, m)
plot_boot("IID Uniform", m, x)

# OU 
for theta in [0.01, 1.5, 10]:
    x = ornstein_uhlenbeck(theta, n)
    m = bootstrap_means(x)
    label = f"OU theta={theta}"
    print_summary(label, x, m)
    plot_boot(label, m, x)

# fGn 
for H in [0.5, 0.75, 0.99999]:
    x = fgn(H, n)
    m = bootstrap_means(x)
    label = f"fGn H={H}"
    print_summary(label, x, m)
    plot_boot(label, m, x)
