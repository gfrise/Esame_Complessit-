import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd

# Impostazioni generali
np.random.seed(42)
sample_size = 1000
n_bootstrap = 1000

def bootstrap_mean(sample, n_bootstrap=1000):
    """Esegue bootstrap con rimpiazzo per stimare la media."""
    means = np.array([
        np.mean(np.random.choice(sample, size=len(sample), replace=True))
        for _ in range(n_bootstrap)
    ])
    return means

def analyze_distribution(dist_name, sample, theoretical_mean, theoretical_std):
    """Analizza la distribuzione e stampa istogrammi e metriche."""
    boot_means = bootstrap_mean(sample, n_bootstrap)
    
    # Metriche bootstrap
    boot_mean = np.mean(boot_means)
    boot_std = np.std(boot_means)

    # Grafici
    plt.figure()
    plt.hist(sample, bins=50, alpha=0.6, density=True)
    plt.title(f"{dist_name} - Campione")
    plt.xlabel("Valore")
    plt.ylabel("Frequenza")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.hist(boot_means, bins=50, alpha=0.7, density=True)
    plt.title(f"{dist_name} - Bootstrap delle medie")
    plt.xlabel("Media")
    plt.ylabel("Frequenza")
    plt.grid(True)
    plt.show()

    # Risultati
    results = {
        "Distribuzione": dist_name,
        "Media teorica": theoretical_mean,
        "Media bootstrap": boot_mean,
        "Errore media": abs(boot_mean - theoretical_mean),
        "Std teorica": theoretical_std,
        "Std bootstrap delle medie": boot_std,
        "Errore std": abs(boot_std - theoretical_std)
    }

    return results

# Lista dei risultati
results = []

# 1. Normale standard
sample = np.random.normal(0, 1, sample_size)
results.append(analyze_distribution("Normale standard", sample, 0, 1 / np.sqrt(sample_size)))

# 2. Student t (df=1.5) - varianza fortemente divergente
df = 1.5
sample = stats.t.rvs(df, size=sample_size)
results.append(analyze_distribution("Student t (df=1.5)", sample, 0, np.nan))  # std teorica non definita

# 3. Student t (df=2.1) - varianza ancora divergente ma meno
df = 2.1
sample = stats.t.rvs(df, size=sample_size)
theo_std = np.sqrt(df / (df - 2)) / np.sqrt(sample_size)
results.append(analyze_distribution("Student t (df=2.1)", sample, 0, theo_std))

# 4. Student t (df=5) - varianza finita
df = 5
sample = stats.t.rvs(df, size=sample_size)
theo_std = np.sqrt(df / (df - 2)) / np.sqrt(sample_size)
results.append(analyze_distribution("Student t (df=5)", sample, 0, theo_std))

# 5. Pareto (α=1.5) - varianza molto divergente
alpha = 1.5
pareto_sample = (np.random.pareto(alpha, sample_size) + 1)
theo_mean = alpha / (alpha - 1) if alpha > 1 else np.inf
results.append(analyze_distribution("Pareto (α=1.5)", pareto_sample, theo_mean, np.nan))

# 6. Pareto (α=2.1) - varianza divergente ma meno
alpha = 2.1
pareto_sample = (np.random.pareto(alpha, sample_size) + 1)
theo_mean = alpha / (alpha - 1)
theo_std = np.sqrt(alpha / ((alpha - 1)**2 * (alpha - 2))) / np.sqrt(sample_size)
results.append(analyze_distribution("Pareto (α=2.1)", pareto_sample, theo_mean, theo_std))

# 7. Pareto (α=3.5) - varianza finita
alpha = 3.5
pareto_sample = (np.random.pareto(alpha, sample_size) + 1)
theo_mean = alpha / (alpha - 1)
theo_std = np.sqrt(alpha / ((alpha - 1)**2 * (alpha - 2))) / np.sqrt(sample_size)
results.append(analyze_distribution("Pareto (α=3.5)", pareto_sample, theo_mean, theo_std))

# Mostra tabella risultati
df_results = pd.DataFrame(results)
import caas_jupyter_tools as tools; tools.display_dataframe_to_user(name="Risultati Bootstrap", dataframe=df_results)

