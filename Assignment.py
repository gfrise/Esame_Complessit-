import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

rng = np.random.default_rng(42)

n = 3000        # sample size
B = 2000        # bootstrap replicates
bins = 60       # histogram bins

def bootstrap_means(x, B=1000, rng=None):
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

def _fmt(x): return f"{x:.4f}" if (x is not None) else "—"

def plot_with_stats(title, means, summary, bins=60):
    plt.figure(figsize=(12, 8))
    plt.hist(means, bins=bins, density=True, alpha=0.7, edgecolor="black")

    mu, sigma = np.mean(means), np.std(means, ddof=1)
    x_vals = np.linspace(mu - 6*sigma, mu + 6*sigma, 200)
    plt.plot(x_vals, norm.pdf(x_vals, mu, sigma*1.2), 'r--', lw=2, alpha=0.6, label="Gaussiana di riferimento")

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
# 2) Lista esperimenti
# ------------------------------
experiments = []

# Normal
x = rng.normal(0,1,n)
m = bootstrap_means(x,B,rng)
experiments.append((m, summarize_bootstrap(m,"Normal(0,1)",original=x)))

# Uniform IID
x = rng.uniform(-2,2,n)
m = bootstrap_means(x,B,rng)
experiments.append((m, summarize_bootstrap(m,"IID Uniform",original=x)))

# t-Student
for nu in [0.4, 1.0, 5.0, 20.0]:
    x = rng.standard_t(df=nu, size=n)
    m = bootstrap_means(x,B,rng)
    desc = f"t-Student(df={nu})"
    if nu < 1: desc += " no mean"
    elif nu < 2: desc += " inf var"
    elif nu < 30: desc += " finite var"
    else: desc += " ~Normal"
    experiments.append((m, summarize_bootstrap(m,desc,original=x)))

# Pareto
def pareto_sample(alpha,size,rng):
    return (1 - rng.random(size))**(-1/alpha)
for alpha in [0.4,1.0,5.0,20.0]:
    x = pareto_sample(alpha,n,rng)
    m = bootstrap_means(x,B,rng)
    desc = f"Pareto(alpha={alpha})"
    if alpha < 1: desc += " no mean"
    elif alpha < 2: desc += " inf var"
    elif alpha < 10: desc += " finite var"
    else: desc += " ~Normal"
    experiments.append((m, summarize_bootstrap(m,desc,original=x)))

# OU
def ornstein_uhlenbeck(n, theta=0.5, mu=0.0, x0=0.0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    x = np.zeros(n); x[0]=x0
    for t in range(1,n):
        x[t] = x[t-1] + theta*(mu-x[t-1]) + rng.normal()
    return x
for theta in [0.01,1.2,2]:
    x = ornstein_uhlenbeck(n,theta=theta,rng=rng)
    m = bootstrap_means(x,B,rng)
    experiments.append((m, summarize_bootstrap(m,f"OU theta={theta}",original=x)))

# fGn
from scipy.linalg import toeplitz, cholesky
def fgn(n, hurst, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    gamma = lambda k: 0.5*((abs(k+1)**(2*hurst) - 2*abs(k)**(2*hurst) + abs(k-1)**(2*hurst)))
    cov = np.array([gamma(k) for k in range(n)])
    L = cholesky(toeplitz(cov), lower=True)
    return L @ rng.normal(size=n)
for H in [0.5,0.75,0.95]:
    x = fgn(n,H,rng)
    m = bootstrap_means(x,B,rng)
    experiments.append((m, summarize_bootstrap(m,f"fGn H={H}",original=x)))

#Plot
for means, summary in experiments:
    plot_with_stats(summary["label"], means, summary, bins) 
    # allora, vedi questo codice? è un assignment che sto facendo per un esame all'università. Vorrei sapere se ci sono altre cose che posso analizzare di facile implementazione ma che aiuterebbero a rendere il progetto più presentabile, sicuramente voglio aggiungere qualcosa sui percentile, su come in casi di std divergente e memoria smettono di essere accurati. POi non so cos'altro di non troppo impegnativo. Inoltre, voglio spostare questo assignment in una nuova cartella dove ci sono due file separati, uno per la divergenza e l'altro per la memoria, quindi vorrei che miseparassi questo file in altri due, non cambiare assolutamente niente della logica, a eccezione del fatto che voglio semplificare il modo in cui vengono plottate. Invece di una sola funzione che plotta tutto, voglio che alla fine di ogni processo avvenga il plot, in maniera così più lunga ma semplice e lineare anche se ridondante, e che gli altri dati del bootstrap vengano solo stampati su console non insieme alla distribuzione 