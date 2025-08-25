import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

# assignment va data 3 giorni prima, vedere quando è programmazione. Cosa succede se non è una dist con varianza finita (quando si calcola l'intervallo di confidenza si assume che sia gaussiana)
# quindi cosa succede quando la dist delle medie non è gaussiana e quando i valori non sono indipendenti, il ruolo della memoria (ou e altro)
rng = np.random.default_rng(42)

n = 20000        # sample size
B = 2000        # bootstrap replicates
bins = 60       # histogram bins

def bootstrap_means(x, B=1000, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    n = len(x)
    idx = rng.integers(0, n, size=(B, n))
    means = x[idx].mean(axis=1)
    return means

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
summaries = []

# 1) IID CASES
iid_datasets = [
    ("Normal(0,1)", rng.normal(0,1,n)),
    ("Uniform(-2,2)", rng.uniform(-2,2,n))
]
# 1a) Normal(0,1)
x_norm = rng.normal(loc=0.0, scale=1.0, size=n)
m_norm = bootstrap_means(x_norm, B=B, rng=rng)
summaries.append(summarize_bootstrap(m_norm, "Normal(0,1)", original=x_norm))

# ===== t-Student =====
t_results = []
for nu in [0.4, 1.0, 5.0, 10.0]:
    x = rng.standard_t(df=nu, size=n)
    m = bootstrap_means(x, B=B, rng=rng)
    desc = f"t-Student(df={nu})"
    if nu < 1:
        desc += " no mean"
    elif nu < 2:
        desc += " inf var"
    elif nu < 30:
        desc += " finite var"
    else:
        desc += " ~Normal"
    summaries.append(summarize_bootstrap(m, desc, original=x))   # <── QUI
    t_results.append((desc, m))

# ===== Pareto =====
xm = 1.0
def pareto_sample(alpha, size, rng):
    return xm * (1 - rng.random(size)) ** (-1.0/alpha)

p_results = []
for alpha in [0.4, 1.2, 5.0, 10.0]:
    x = pareto_sample(alpha, n, rng)
    m = bootstrap_means(x, B=B, rng=rng)
    desc = f"Pareto(alpha={alpha})"
    if alpha < 1:
        desc += " no mean"
    elif alpha < 2:
        desc += " inf var"
    elif alpha < 10:
        desc += " finite var"
    else:
        desc += " ~Normal"
    summaries.append(summarize_bootstrap(m, desc, original=x))   # <── QUI
    p_results.append((desc, m))

# costruzione finale di datasets (presuppone m_norm già definito)
datasets = [("Normal(0,1)", m_norm)] + t_results + p_results

def _fmt(x): return f"{x:.4f}" if (x is not None) else "—"

def plot_with_stats(title, means, summary, bins=60):
    plt.figure(figsize=(12, 8))

    # Istogramma bootstrap
    plt.hist(means, bins=bins, density=True, alpha=0.7, edgecolor="black")

    # Gaussiana di riferimento
    mu = np.mean(means)
    sigma = np.std(means, ddof=1)
    x_vals = np.linspace(mu - 6*sigma, mu + 6*sigma, 200)
    plt.plot(
        x_vals, norm.pdf(x_vals, mu, sigma*1.2),
        'r--', lw=2, alpha=0.6, label="Gaussiana di riferimento"
    )

    # Estraggo valori dal dizionario
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

    stats_text = "\n".join([f"{k:<20} {_fmt(v)}" for k, v in rows])

    plt.text(
        0.02, 0.98, stats_text,
        transform=plt.gca().transAxes,
        fontsize=11,
        fontfamily='monospace',
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='black', alpha=0.9)
    )

    plt.title(f"Bootstrap means — {title}\n")
    plt.xlabel("bootstrap means")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Modifica i cicli di plotting per utilizzare la nuova funzione
for title, means in datasets:
    # Trova la summary corrispondente
    summary = next((s for s in summaries if s['label'] == title), None)
    if summary:
        plot_with_stats(title, means, summary, bins)
# ------------------------------
# 1) IID: Rumore bianco gaussiano
x_iid = rng.uniform(-2,2,size=n)
m_iid = bootstrap_means(x_iid, B=B, rng=rng)
summaries.append(summarize_bootstrap(m_iid, "IID Uniform", original=x_iid))

# ------------------------------
# ------------------------------
# 2) Memoria breve: Ornstein-Uhlenbeck
def ornstein_uhlenbeck(n, theta=0.5, sigma=1.0, mu=0.0, x0=0.0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    dt = 1.0
    x = np.zeros(n)
    x[0] = x0
    for t in range(1, n):
        x[t] = x[t-1] + theta*(mu - x[t-1])*dt + sigma*np.sqrt(dt)*rng.normal()
    return x

theta_list = [0.05, 0.5, 1.0, 2]  # variazioni di theta
datasets_mem = []  # raccoglie tutti i bootstrap per plottarli

for theta in theta_list:
    x_ou = ornstein_uhlenbeck(n, theta=theta, sigma=1.0, rng=rng)
    m_ou = bootstrap_means(x_ou, B=B, rng=rng)
    label = f"OU theta={theta}"
    # solo una summary, con original
    summaries.append(summarize_bootstrap(m_ou, label, original=x_ou))
    datasets_mem.append((label, m_ou))


# ------------------------------
# 3) Memoria lunga: Fractional Gaussian Noise (H=0.8)
def fgn(n, hurst=0.8, rng=None):
    """Hosking’s method (semplificata) per generare fractional Gaussian noise"""
    rng = np.random.default_rng() if rng is None else rng
    gamma = lambda k: 0.5*((abs(k+1)**(2*hurst) - 2*abs(k)**(2*hurst) + abs(k-1)**(2*hurst)))
    cov = np.array([gamma(k) for k in range(n)])
    from scipy.linalg import toeplitz, cholesky
    C = toeplitz(cov)
    L = cholesky(C, lower=True)
    z = rng.normal(size=n)
    return L @ z   # ritorna solo il segnale

hurst_list = [0.1, 0.3, 0.5, 0.6, 0.8, 0.95]  # variazioni di H

for H in hurst_list:
    x_fgn = fgn(n, hurst=H, rng=rng)
    m_fgn = bootstrap_means(x_fgn, B=B, rng=rng)
    label = f"fGn H={H}"
    # solo una summary, con original
    summaries.append(summarize_bootstrap(m_fgn, label, original=x_fgn))
    datasets_mem.append((label, m_fgn))

# ------------------------------
# PLOT
datasets = [("IID Uniform", m_iid)] + datasets_mem

for title, means in datasets:
    summary = next((s for s in summaries if s['label'] == title), None)
    if summary:
        plot_with_stats(title, means, summary, bins)