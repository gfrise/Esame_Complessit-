# assignment va data 3 giorni prima, vedere quando è programmazione. Cosa succede se non è una dist con varianza finita (quando si calcola l'intervallo di confidenza si assume che sia gaussiana)
# quindi cosa succede quando la dist delle medie non è gaussiana e quando i valori non sono indipendenti, il ruolo della memoria (ou e altro)

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
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

# 1b) t-Student finite variance (nu=5)
nu_finite = 5.0
x_t5 = rng.standard_t(df=nu_finite, size=n)
m_t5 = bootstrap_means(x_t5, B=B, rng=rng)
summaries.append(summarize_bootstrap(m_t5, f"t-Student(df={nu_finite}) finite var"))

# 1c) t-Student infinite variance (nu=1.5) -> variance diverges, mean exists (nu>1)
nu_inf = 1.5
x_t15 = rng.standard_t(df=nu_inf, size=n)
m_t15 = bootstrap_means(x_t15, B=B, rng=rng)
summaries.append(summarize_bootstrap(m_t15, f"t-Student(df={nu_inf}) infinite var"))

# 1d) Power-law / Pareto Type I with xm=1
xm = 1.0

def pareto_sample(alpha, size, rng):
    # Pareto Type I with scale xm: X = xm * (1 - U)^(-1/alpha), U~U(0,1)
    U = rng.random(size)
    return xm * (1 - U) ** (-1.0/alpha)

# convergent variance: alpha=3 (>2)
alpha_conv = 3.0
x_pl_conv = pareto_sample(alpha_conv, n, rng)
m_pl_conv = bootstrap_means(x_pl_conv, B=B, rng=rng)
summaries.append(summarize_bootstrap(m_pl_conv, f"Pareto(alpha={alpha_conv}) finite var"))

# divergent variance: alpha=1.5 (<=2)
alpha_div = 1.5
x_pl_div = pareto_sample(alpha_div, n, rng)
m_pl_div = bootstrap_means(x_pl_div, B=B, rng=rng)
summaries.append(summarize_bootstrap(m_pl_div, f"Pareto(alpha={alpha_div}) infinite var"))

# Show histograms for the bootstrap means
datasets = [
    ("Normal(0,1)", m_norm),
    (f"t(df={nu_finite}) finite var", m_t5),
    (f"t(df={nu_inf}) infinite var", m_t15),
    (f"Pareto(alpha={alpha_conv}) finite var", m_pl_conv),
    (f"Pareto(alpha={alpha_div}) infinite var", m_pl_div),
]

for title, means in datasets:
    plt.figure()
    plt.hist(means, bins=bins, density=True, edgecolor="black")
    plt.title(f"Bootstrap means — {title}\n(n={n}, B={B})")
    plt.xlabel("mean* (bootstrap)")
    plt.ylabel("density")
    plt.tight_layout()
    plt.show()

df_sum = pd.DataFrame(summaries)

from caas_jupyter_tools import display_dataframe_to_user
display_dataframe_to_user("Bootstrap means — IID cases: summary stats (ripristinato)", df_sum)

csv_path = "/mnt/data/bootstrap_iid_summary.csv"
df_sum.to_csv(csv_path, index=False)
csv_path
