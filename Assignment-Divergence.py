import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

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
    print(f"Mean (boot)          {np.mean(m):.3f}")
    print(f"Std (original)       {sd:.3f}")
    print(f"Std (boot)            {np.std(m,ddof=1):.3f}")
    print(f"Std theoretical      {sd/np.sqrt(len(x)):.3f}")
    print(f"2.5%                 {np.percentile(m,2.5):.3f}")
    print(f"97.5%                {np.percentile(m,97.5):.3f}")
    print(f"Skew                 {stats.skew(m):.3f}")
    print(f"Kurt (ex)        {stats.kurtosis(m):.3f}")


# Plot
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

    #stats
    rows = [
        ("Mean (boot)", mu),
        ("SE (boot)", sd),
        ("2.5%", p1),
        ("97.5%", p2),
        ("Skew", stats.skew(m)),
        ("Kurtosis (ex)", stats.kurtosis(m)),
    ]

    if x is not None:
        sx = np.std(x, ddof=1)
        rows = [
            ("Mean (orig)", np.mean(x)),
            ("Std (orig)", sx),
            ("Std theor", sx/np.sqrt(len(x))),
        ] + rows

    txt = "\n".join(f"{k:<14} {v:.4f}" for k,v in rows)

    plt.text(0.02, 0.98, txt,
             transform=plt.gca().transAxes,
             va='top', ha='left',
             fontsize=11, family='monospace',
             bbox=dict(boxstyle='round', fc='white', ec='black', alpha=0.9))


    plt.title(title)
    plt.xlabel("bootstrap means")
    plt.ylabel("density")
    plt.tight_layout()
    plt.show()


# EXPERIMENTS
# Normal
x = rng.normal(0,1,n)
m = bootstrap_means(x)
print_summary("Normal(0,1)", x, m)
plot_boot("Normal(0,1)", m, x)

# Student-t
for nu in [0.4, 1.0, 5.0, 20.0]:
    x = rng.standard_t(nu, n)
    m = bootstrap_means(x)

    label = f"t-Student(df={nu})"
    if nu < 1: label += " no mean"
    elif nu < 2: label += " inf var"
    elif nu < 15: label += " finite var"
    else: label += " ~Normal"

    print_summary(label, x, m)
    plot_boot(label, m, x)

# Pareto
def pareto(a):
    return (1 - rng.random(n))**(-1/a)

for a in [0.4, 1.0, 5.0, 20.0]:
    x = pareto(a)
    m = bootstrap_means(x)

    label = f"Pareto(alpha={a})"
    if a < 1: label += " no mean"
    elif a < 2: label += " inf var"
    elif a < 15: label += " finite var"
    else: label += " ~Normal"

    print_summary(label, x, m)
    plot_boot(label, m, x)
