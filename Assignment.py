# assignment va data 3 giorni prima, vedere quando è programmazione. Cosa succede se non è una dist con varianza finita (quando si calcola l'intervallo di confidenza si assume che sia gaussiana)
# quindi cosa succede quando la dist delle medie non è gaussiana e quando i valori non sono indipendenti, il ruolo della memoria (ou e altro)

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
import pandas as pd

rng = np.random.default_rng(42)

n = 1000        # sample size
B = 2000        # bootstrap replicates
bins = 60       # histogram bins

def bootstrap_means(x, B=1000, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    n = len(x)
    idx = rng.integers(0, n, size=(B, n))
    means = x[idx].mean(axis=1)
    return means

def summarize_bootstrap(means, label):
    return {
        "label": label,
        "boot_mean_of_means": float(np.mean(means)),
        "boot_se": float(np.std(means, ddof=1)),
        "p2.5": float(np.percentile(means, 2.5)),
        "p97.5": float(np.percentile(means, 97.5)),
        "skew": float(stats.skew(means)),
        "kurtosis_excess": float(stats.kurtosis(means)),  # Fisher, 0=normal
    }

summaries = []

# 1) IID CASES

# 1a) Normal(0,1)
x_norm = rng.normal(loc=0.0, scale=1.0, size=n)
m_norm = bootstrap_means(x_norm, B=B, rng=rng)
summaries.append(summarize_bootstrap(m_norm, "Normal(0,1)"))

# ===== t-Student =====
# Caso 1: nu=0.8 (media e varianza non esistono)
nu08 = 0.8
x_t08 = rng.standard_t(df=nu08, size=n)
m_t08 = bootstrap_means(x_t08, B=B, rng=rng)
summaries.append(summarize_bootstrap(m_t08, f"t-Student(df={nu08}) no mean"))

# Caso 2: nu=1.5 (media finita, varianza infinita)
nu15 = 1.5
x_t15 = rng.standard_t(df=nu15, size=n)
m_t15 = bootstrap_means(x_t15, B=B, rng=rng)
summaries.append(summarize_bootstrap(m_t15, f"t-Student(df={nu15}) inf var"))

# Caso 3: nu=5 (media e var finite, heavy tails)
nu5 = 5.0
x_t5 = rng.standard_t(df=nu5, size=n)
m_t5 = bootstrap_means(x_t5, B=B, rng=rng)
summaries.append(summarize_bootstrap(m_t5, f"t-Student(df={nu5}) finite var"))

# Caso 4: nu=30 (quasi Normale)
nu30 = 30.0
x_t30 = rng.standard_t(df=nu30, size=n)
m_t30 = bootstrap_means(x_t30, B=B, rng=rng)
summaries.append(summarize_bootstrap(m_t30, f"t-Student(df={nu30}) ~Normal"))

# ===== Pareto =====
xm = 1.0
def pareto_sample(alpha, size, rng):
    U = rng.random(size)
    return xm * (1 - U) ** (-1.0/alpha)

# Caso A: alpha=0.8 → mean e var divergenza
alpha08 = 0.8
x_pl08 = pareto_sample(alpha08, n, rng)
m_pl08 = bootstrap_means(x_pl08, B=B, rng=rng)
summaries.append(summarize_bootstrap(m_pl08, f"Pareto(alpha={alpha08}) no mean"))

# Caso B: alpha=1.5 → mean esiste, var infinita
alpha15 = 1.5
x_pl15 = pareto_sample(alpha15, n, rng)
m_pl15 = bootstrap_means(x_pl15, B=B, rng=rng)
summaries.append(summarize_bootstrap(m_pl15, f"Pareto(alpha={alpha15}) inf var"))

# Caso C: alpha=3 → mean e var finite
alpha3 = 3.0
x_pl3 = pareto_sample(alpha3, n, rng)
m_pl3 = bootstrap_means(x_pl3, B=B, rng=rng)
summaries.append(summarize_bootstrap(m_pl3, f"Pareto(alpha={alpha3}) finite var"))

# Caso D: alpha=10 → molto vicina a Normale
alpha10 = 10.0
x_pl10 = pareto_sample(alpha10, n, rng)
m_pl10 = bootstrap_means(x_pl10, B=B, rng=rng)
summaries.append(summarize_bootstrap(m_pl10, f"Pareto(alpha={alpha10}) ~Normal"))

datasets = [
    ("Normal(0,1)", m_norm),
    (f"t(df={nu08}) no mean", m_t08),
    (f"t(df={nu15}) inf var", m_t15),
    (f"t(df={nu5}) finite var", m_t5),
    (f"t(df={nu30}) ~Normal", m_t30),
    (f"Pareto(alpha={alpha08}) no mean", m_pl08),
    (f"Pareto(alpha={alpha15}) inf var", m_pl15),
    (f"Pareto(alpha={alpha3}) finite var", m_pl3),
    (f"Pareto(alpha={alpha10}) ~Normal", m_pl10),
]


for title, means in datasets:
    plt.figure()
    
    # istogramma bootstrap
    plt.hist(means, bins=bins, density=True, alpha=0.7, edgecolor="black")
    
    # gaussiana di riferimento (media e sigma dal bootstrap)
    mu = np.mean(means)
    sigma = np.std(means, ddof=1)
    x_vals = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
    
    # gaussiana smussata per evidenziare le differenze
    plt.plot(x_vals, norm.pdf(x_vals, mu, sigma*1.2), 'r--', lw=2, alpha=0.6, label="Gaussiana di riferimento")
    
    plt.title(f"Bootstrap means — {title}\n(n={n}, B={B})")
    plt.xlabel("bootstrap means")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.show()
# df_sum = pd.DataFrame(summaries)

# from caas_jupyter_tools import display_dataframe_to_user
# display_dataframe_to_user("Bootstrap means — IID cases: summary stats (ripristinato)", df_sum)

# csv_path = "/mnt/data/bootstrap_iid_summary.csv"
# df_sum.to_csv(csv_path, index=False)
# csv_path


# ------------------------------
# 1) IID: Rumore bianco gaussiano
# x_iid = rng.normal(0, 1, size=n)
x_iid = rng.uniform(-2,2,size=n)
m_iid = bootstrap_means(x_iid, B=B, rng=rng)
# summaries.append(summarize_bootstrap(m_iid, "IID Normal"))
summaries.append(summarize_bootstrap(m_iid, "IID Uniform"))


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

theta_list = [0.1, 0.5, 1.0, 2]  # variazioni di theta
datasets_mem = []  # raccoglie tutti i bootstrap per plottarli

for theta in theta_list:
    x_ou = ornstein_uhlenbeck(n, theta=theta, sigma=1.0, rng=rng)
    m_ou = bootstrap_means(x_ou, B=B, rng=rng)
    label = f"OU theta={theta}"
    summaries.append(summarize_bootstrap(m_ou, label))
    datasets_mem.append((label, m_ou))

# ------------------------------
# 3) Memoria lunga: Fractional Gaussian Noise (H=0.8)
def fgn(n, hurst=0.8, rng=None):
    """Hosking’s method (semplificata) per generare fractional Gaussian noise"""
    rng = np.random.default_rng() if rng is None else rng
    gamma = lambda k: 0.5*((abs(k+1)**(2*hurst) - 2*abs(k)**(2*hurst) + abs(k-1)**(2*hurst)))
    # autocovarianza
    cov = np.array([gamma(k) for k in range(n)])
    from scipy.linalg import toeplitz, cholesky
    C = toeplitz(cov)
    L = cholesky(C, lower=True)
    z = rng.normal(size=n)
    return L @ z

hurst_list = [0.1, 0.3, 0.5, 0.6, 0.8, 0.95]  # variazioni di H

for H in hurst_list:
    x_fgn = fgn(n, hurst=H, rng=rng)
    m_fgn = bootstrap_means(x_fgn, B=B, rng=rng)
    label = f"fGn H={H}"
    summaries.append(summarize_bootstrap(m_fgn, label))
    datasets_mem.append((label, m_fgn))

# ------------------------------
# PLOT
datasets = [("IID Uniform", m_iid)] + datasets_mem


for title, means in datasets:
    plt.figure()
    
    # istogramma bootstrap
    plt.hist(means, bins=bins, density=True, alpha=0.7, edgecolor="black")
    
    # gaussiana di riferimento (media e sigma dal bootstrap)
    mu = np.mean(means)
    sigma = np.std(means, ddof=1)
    x_vals = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
    
    # gaussiana smussata per evidenziare le differenze
    plt.plot(x_vals, norm.pdf(x_vals, mu, sigma*1.2), 'r--', lw=2, alpha=0.6, label="Gaussiana di riferimento")
    
    plt.title(f"Bootstrap means — {title}\n(n={n}, B={B})")
    plt.xlabel("bootstrap means")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ------------------------------
# Summary table
df_sum = pd.DataFrame(summaries)
print(df_sum)

# aggiungere dati come media e sd stimata e reale, in jupyter o py? 
