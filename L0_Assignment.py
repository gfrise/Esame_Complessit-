import numpy as np, matplotlib.pyplot as plt
from scipy.stats import t,pareto

N, i, M, SD, df, alpha = 10**3, 1000, 3, 0.4, 5, 1.5

def OU(N,M,SD,theta,dt=1):
    x = np.empty(N)
    x[0] = M
    for t in range(1, N):
        x[t] = x[t-1] + theta*(M - x[t-1])*dt + SD*np.sqrt(dt)*np.random.randn()
    return x

# means_t = [np.mean(t.rvs(df=df, loc=M, scale=SD/np.sqrt(df/(df-2)), size=N)) for _ in range(i)]
# mean_pl = [np.mean(pareto.rvs(alpha, size=N)) for _ in range(i)]
means_gauss = [np.mean(np.random.normal(M,SD,N)) for _ in range(i)]
means_ou_low = [np.mean(OU(N,M,SD,theta=1.5)) for _ in range(i)]   #θ grande poca memoria
means_ou_high= [np.mean(OU(N,M,SD,theta=0.001)) for _ in range(i)]  #θ piccolo molta memoria

def plotta(data, title, M, SD, N):
    emp_mean = np.mean(data)
    emp_std = np.std(data)*np.sqrt(N)

    plt.hist(data, bins=30, edgecolor="black")
    plt.title(
        f"{title}\n"
        f"μ bootstrap = {emp_mean:.3f}   σ bootstrap = {emp_std:.3f}\n"
        f"μ teorica = {M:.3f}          σ teorica = {SD:.3f}"
    )    
    plt.xlabel("Media campionaria")
    plt.ylabel("Frequenza")
    plt.show()

plotta(means_gauss, "iid Gaussiana", M, SD, N)
plotta(means_ou_low, "OU memoria breve", M, SD, N)
plotta(means_ou_high, "OU memoria lunga", M, SD, N)
