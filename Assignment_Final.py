import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

# ------------------------------
# Setup
# ------------------------------
rng = np.random.default_rng(42)

n = 3000      # sample size
B = 3000      # bootstrap replicates
bins = 60     # histogram bins


# ------------------------------
# Bootstrap core
# ------------------------------
def bootstrap_means(x, B, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    n = len(x)
    idx = rng.integers(0, n, size=(B, n))
    return x[idx].mean(axis=1)


# ------------------------------
# Summary
# ------------------------------
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


# ------------------------------
# Console print
# ------------------------------
def print_summary(summary):
    def fmt(v):
        return f"{v:.4f}" if v is not None else "—"

    keys = [
        ("original_mean",   "Mean (original)"),
        ("original_std",    "Std (original)"),
        ("boot_mean_of_means", "Mean (boot)"),
        ("boot_se",         "SE (boot)"),
        ("p2.5",            "2.5%"),
        ("p97.5",           "97.5%"),
        ("skew",            "Skew"),
        ("kurtosis_excess", "Kurtosis (ex)"),
        ("std_theoretical", "Std theoretical")
    ]

    print(f"\n--- {summary['label']} ---")
    for key, label in keys:
        print(f"{label:<20} {fmt(summary.get(key))}")


# ------------------------------
# Histogram plot
# ------------------------------
def plot_with_stats(title, means, summary, bins=60):
    plt.figure(figsize=(12, 8))
    plt.hist(means, bins=bins, density=True, alpha=0.7, edgecolor="black")

    mu, sigma = np.mean(means), np.std(means, ddof=1)
    x_vals = np.linspace(mu - 6*sigma, mu + 6*sigma, 300)
    plt.plot(x_vals, norm.pdf(x_vals, mu, sigma),
             'r--', lw=2, alpha=0.7, label="Gaussian reference")

    # Percentiles
    p_low = summary["p2.5"]
    p_high = summary["p97.5"]
    plt.axvline(p_low, color='orange', linestyle='--', lw=2)
    plt.axvline(p_high, color='orange', linestyle='--', lw=2)

    # Text box
    rows = [
        ("Mean (boot)", summary["boot_mean_of_means"]),
        ("SE (boot)", summary["boot_se"]),
        ("2.5%", p_low),
        ("97.5%", p_high),
        ("Skew", summary["skew"]),
        ("Kurtosis", summary["kurtosis_excess"]),
    ]
    if "std_theoretical" in summary:
        rows.append(("Std theoretical", summary["std_theoretical"]))

    text = "\n".join([f"{k:<20} {v:.4f}" for k, v in rows])

    plt.text(0.02, 0.98, text,
             transform=plt.gca().transAxes,
             va="top", ha="left",
             fontsize=11, family="monospace",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    plt.title(f"Bootstrap means — {title}")
    plt.xlabel("bootstrap means")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ------------------------------
# Diagnostic metrics
# ------------------------------
def diagnostics(summary, true_mean=None):
    out = {}

    if true_mean is not None:
        out["mean_bias"] = summary["boot_mean_of_means"] - true_mean
        out["coverage"] = int(summary["p2.5"] <= true_mean <= summary["p97.5"])

    if "std_theoretical" in summary:
        out["se_ratio"] = summary["boot_se"] / summary["std_theoretical"]

    out["skew"] = summary["skew"]
    out["kurtosis"] = summary["kurtosis_excess"]
    out["ci_width"] = summary["p97.5"] - summary["p2.5"]

    return out


# ------------------------------
# Diagnostic plot
# ------------------------------
def plot_diagnostics(diags, labels):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    metrics = [
        ("mean_bias", "Bias of mean"),
        ("se_ratio", "SE / SE theoretical"),
        ("skew", "Skewness"),
        ("kurtosis", "Kurtosis (excess)"),
        ("ci_width", "CI width"),
        ("coverage", "Coverage (0/1)")
    ]

    for ax, (key, title) in zip(axes, metrics):
        values = [d.get(key, np.nan) for d in diags]
        ax.plot(values, marker="o")
        ax.axhline(0, color="black", lw=1)
        ax.set_title(title)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45)

    plt.suptitle("Bootstrap reliability diagnostics", fontsize=14)
    plt.tight_layout()
    plt.show()


# ------------------------------
# Experiments
# ------------------------------
experiments = []
diagnostics_list = []
labels = []

# Normal
x = rng.normal(0, 1, n)
m = bootstrap_means(x, B, rng)
s = summarize_bootstrap(m, "Normal(0,1)", original=x)
experiments.append((m, s))
diagnostics_list.append(diagnostics(s, true_mean=0.0))
labels.append("Normal")

# t-Student
for nu in [0.4, 1.0, 5.0, 20.0]:
    x = rng.standard_t(df=nu, size=n)
    m = bootstrap_means(x, B, rng)
    label = f"t(df={nu})"
    s = summarize_bootstrap(m, label, original=x)
    experiments.append((m, s))
    diagnostics_list.append(diagnostics(s, true_mean=0.0))
    labels.append(label)

# Pareto
def pareto_sample(alpha, size, rng):
    return (1 - rng.random(size)) ** (-1 / alpha)

for alpha in [0.4, 1.0, 5.0, 20.0]:
    x = pareto_sample(alpha, n, rng)
    m = bootstrap_means(x, B, rng)
    label = f"Pareto(alpha={alpha})"
    s = summarize_bootstrap(m, label, original=x)
    experiments.append((m, s))
    diagnostics_list.append(diagnostics(s, true_mean=None))
    labels.append(label)


# ------------------------------
# Run
# ------------------------------
for means, summary in experiments:
    print_summary(summary)
    plot_with_stats(summary["label"], means, summary, bins)

plot_diagnostics(diagnostics_list, labels)
