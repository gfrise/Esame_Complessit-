# coverage_strip_and_metrics.py
"""
Minimo ma completo:
- esegue R repliche per un parametro
- costruisce percentile bootstrap CI della mean
- calcola coverage, MC-SE, avg_width, bias, RMSE, interval_score, ks p-value sui p-value
- produce lo strip-plot (come immagine che hai mostrato)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, kstest
from scipy.linalg import toeplitz, cholesky

# ---------------- user params ----------------
seed = 123
rng = np.random.default_rng(seed)
n = 80           # sample size per replicate
B = 100         # bootstrap replicates per sample
R = 120          # number of independent replicates in strip plot
alpha_ci = 0.05  # CI level 95%
# ---------------------------------------------

# --- example generator: normal with mean=47, sd=5 ---
true_mean = 47.0
def gen_normal(mu, rng, n):
    return rng.normal(mu, 5.0, size=n)

# --- bootstrap percentile CI for mean
def bootstrap_percentile_ci_mean(x, B, rng, alpha=0.05):
    n = len(x)
    idx = rng.integers(0, n, size=(B, n))
    rep_means = x[idx].mean(axis=1)
    lo = np.percentile(rep_means, 100*(alpha/2))
    hi = np.percentile(rep_means, 100*(1-alpha/2))
    return lo, hi, rep_means

# --- interval score
def interval_score(lo, hi, true, alpha):
    width = hi - lo
    score = width
    if true < lo:
        score += 2.0/alpha * (lo - true)
    elif true > hi:
        score += 2.0/alpha * (true - hi)
    return score

# --- wrapper that runs R replicates and returns metrics + arrays for plotting
def run_replicates_and_metrics(generator, param, true_value, n, B, R, rng, alpha_ci):
    lo_arr = np.empty(R)
    hi_arr = np.empty(R)
    theta_arr = np.empty(R)
    pvals = np.empty(R)
    scores = np.empty(R)
    captured = np.zeros(R, dtype=bool)

    for i in range(R):
        x = generator(param, rng, n)
        lo, hi, reps = bootstrap_percentile_ci_mean(x, B, rng, alpha=alpha_ci)
        th = x.mean()
        lo_arr[i] = lo
        hi_arr[i] = hi
        theta_arr[i] = th
        captured[i] = (lo <= true_value <= hi)
        # two-sided bootstrap p-value approximation
        prop_less = np.mean(reps <= true_value)
        p = 2.0 * min(prop_less, 1.0 - prop_less)
        pvals[i] = p
        scores[i] = interval_score(lo, hi, true_value, alpha_ci)

    # metrics
    coverage = captured.mean()
    mc_se = np.sqrt(coverage*(1-coverage)/R)
    avg_width = np.mean(hi_arr - lo_arr)
    bias = np.mean(theta_arr - true_value)
    rmse = np.sqrt(np.mean((theta_arr - true_value)**2))
    avg_score = np.mean(scores)
    # KS test for p-values uniformity
    ks_stat, ks_p = kstest(pvals, 'uniform')

    results = {
        'coverage': coverage,
        'mc_se': mc_se,
        'avg_width': avg_width,
        'bias': bias,
        'rmse': rmse,
        'avg_score': avg_score,
        'ks_p': ks_p,
        'lo_arr': lo_arr, 'hi_arr': hi_arr, 'theta_arr': theta_arr, 'captured': captured
    }
    return results

# --- plotting strip similar to your example ---
def plot_strip(lo_arr, hi_arr, theta_arr, captured, true_value, title=None):
    R = len(lo_arr)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(R):
        color = '#17becf' if captured[i] else '#e45756'  # teal if captured, red if not
        ax.hlines(i, lo_arr[i], hi_arr[i], color=color, linewidth=1.8, alpha=0.9)
        ax.plot(theta_arr[i], i, 'o', color=color, markersize=4)
    ax.axvline(true_value, color='black', linewidth=2)
    ax.set_yticks([])
    ax.set_xlabel('Estimate / CI')
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()

# ----------------- run -----------------
if __name__ == '__main__':
    # pick a parameter (for normal, param is mean)
    param = true_mean
    res = run_replicates_and_metrics(gen_normal, param, true_mean, n, B, R, rng, alpha_ci)

    # print metrics
    print("Coverage: {:.3f} ± {:.3f}".format(res['coverage'], res['mc_se']))
    print("Avg width: {:.3f}".format(res['avg_width']))
    print("Bias: {:.3f}".format(res['bias']))
    print("RMSE: {:.3f}".format(res['rmse']))
    print("Interval score (avg): {:.3f}".format(res['avg_score']))
    print("KS p-value for p-value uniformity: {:.3f}".format(res['ks_p']))

    # plot strip
    plot_strip(res['lo_arr'], res['hi_arr'], res['theta_arr'], res['captured'], true_mean,
               title=f'Strip CIs (n={n}, B={B}, R={R})')
