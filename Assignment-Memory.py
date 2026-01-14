import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import toeplitz, cholesky
from scipy.stats import norm

rng = np.random.default_rng(482)

n = 3000     # sample size
B = 3000      # bootstrap replicates
bins = 60       # histogram bins

def bootstrap_means(x, B, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    n = len(x)
    idx = rng.integers(0, n, size=(B, n))
    return x[idx].mean(axis=1)

def summarize_bootstrap(means, label, original=None, population=None):
    boot_mean = float(np.mean(means))
    boot_std = float(np.std(means, ddof=1))
    summary = {
        "label": label,
        "boot_mean_of_means": boot_mean,
        "boot_se": boot_std,
        "p2.5": float(np.percentile(means, 2.5)),
        "p97.5": float(np.percentile(means, 97.5)),
        "skew": float(stats.skew(means)),
        "kurtosis_excess": float(stats.kurtosis(means)),
    }
    if original is not None:
        l = len(original)
        sd = np.std(original, ddof=1)
        summary.update({
            "original_mean": float(np.mean(original)),
            "original_std": float(sd),
            "std_theoretical": float(sd / np.sqrt(l))
        })
    if population is not None:
        summary.update({
            "population_mean": float(np.mean(population)),
            "population_std": float(np.std(population, ddof=0))
        })
    return summary

# Stampa ordinata su console  
def print_summary(summary):
    def fmt(v):
        return f"{v:.3f}" if v is not None else "—"

    keys = [
        ("original_mean", "Mean (original)"),
        ("original_std", "Std (original)"),
        ("boot_mean_of_means", "Mean (boot)"),
        ("boot_se", "SE (boot)"),
        ("p2.5", "2.5%"),
        ("p97.5", "97.5%"),
        ("skew", "Skew"),
        ("kurtosis_excess", "Kurtosis (ex)"),
        ("std_theoretical", "Std theoretical")
    ]

    print(f"\n--- {summary['label']} ---")
    for key, label in keys:
        print(f"{label:<20} {fmt(summary.get(key))}")

# Plot 
def _fmt(x): return f"{x:.4f}" if (x is not None) else "—"

def plot_with_stats(title, means, summary, bins=60):
    plt.figure(figsize=(12, 8))
    plt.hist(means, bins=bins, density=True, alpha=0.7, edgecolor="black")

    mu, sigma = np.mean(means), np.std(means, ddof=1)
    x_vals = np.linspace(mu - 6*sigma, mu + 6*sigma, 200)
    plt.plot(x_vals, norm.pdf(x_vals, mu, sigma*1.2), 'r--', lw=2, alpha=0.6, label="Gaussiana di riferimento")

    # percentile lines (2.5% and 97.5%)
    p_low = np.percentile(means, 2.5)
    p_high = np.percentile(means, 97.5)
    plt.axvline(p_low, color='orange', linestyle='--', lw=2, label='2.5% percentile')
    plt.axvline(p_high, color='orange', linestyle='--', lw=2, label='97.5% percentile')

    # optional: annotate percentile values slightly above x-axis
    ymin, ymax = plt.gca().get_ylim()
    y_annot = ymax * 0.08  # place annotation a bit above the x-axis
    plt.text(p_low, y_annot, f"{p_low:.4f}", rotation=90, va='bottom', ha='right', color='orange', fontsize=9, family='monospace')
    plt.text(p_high, y_annot, f"{p_high:.4f}", rotation=90, va='bottom', ha='left', color='orange', fontsize=9, family='monospace')

    rows = []
    if summary.get("population_mean") is not None:
        rows.append(("Mean (population)", summary["population_mean"]))
        rows.append(("Std (population)", summary["population_std"]))
    if summary.get("original_mean") is not None:
        rows.append(("Mean (original)", summary["original_mean"]))
        rows.append(("Std (original)", summary["original_std"]))
    rows += [
        ("Mean (boot)", summary.get("boot_mean_of_means")),
        ("SE (boot)", summary.get("boot_se")),
        ("2.5%", summary.get("p2.5")),
        ("97.5%", summary.get("p97.5")),
        ("Skew", summary.get("skew")),
        ("Kurtosis (ex)", summary.get("kurtosis_excess")),
    ]
    if summary.get("std_theoretical") is not None:
        rows.append(("Std theoretical", summary["std_theoretical"]))

    stats_text = "\n".join([f"{k:<20} {_fmt(v)}" for k,v in rows])
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             fontsize=11, fontfamily="monospace",
             va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.6",
                       facecolor="white", edgecolor="black", alpha=0.9))
    plt.title(f"Bootstrap means — {title}\n")
    plt.xlabel("bootstrap means")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ------------------------------  
# Processi con memoria  
# ------------------------------  
def ornstein_uhlenbeck(n, theta=0.5, mu=0.0, x0=0.0, rng=None):
    x = np.zeros(n)
    x[0] = x0
    for t in range(1, n):
        x[t] = x[t-1] + theta*(mu-x[t-1]) + rng.normal()
    return x

def fgn(n, hurst, rng=None):
    gamma = lambda k: 0.5*((abs(k+1)**(2*hurst) - 2*abs(k)**(2*hurst) + abs(k-1)**(2*hurst)))
    cov = np.array([gamma(k) for k in range(n)])
    L = cholesky(toeplitz(cov), lower=True)
    return L @ rng.normal(size=n)

experiments = []

# Uniform IID
x = rng.uniform(-2,2,n)
m = bootstrap_means(x,B,rng)
experiments.append((m, summarize_bootstrap(m,"IID Uniform",original=x)))
# OU
for theta in [0.01, 1.2, 2]:
    x = ornstein_uhlenbeck(n, theta=theta, rng=rng)
    m = bootstrap_means(x, B, rng)
    experiments.append((m, summarize_bootstrap(m, f"OU theta={theta}", original=x)))

# fGn
for H in [0.00001, 0.5, 0.75, 0.99999]:
    x = fgn(n, H, rng)
    m = bootstrap_means(x, B, rng)
    experiments.append((m, summarize_bootstrap(m, f"fGn H={H}", original=x)))

# ------------------------------  
# Esegui esperimenti  
# ------------------------------  
for means, summary in experiments:
    print_summary(summary)             # stampa ordinata su console
    # chiamata corretta alla funzione di plotting che usa summary
    plot_with_stats(summary["label"], means, summary, bins)

