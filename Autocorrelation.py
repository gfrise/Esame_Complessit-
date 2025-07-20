import numpy as np, matplotlib.pyplot as plt
import random

t, n, m, y, tmax = 100, 10**3, 50, 0.1, 50
dt, nn = 1/t, n*t

def AC(x,t):
    m1, m2, s1, s2, cov = 0,0,0,0,0
    rng = len(x)-tmax

    for i in range(rng):
        m1+=x[i]
        m2+=x[i+t]
        s1+=x[i]**2
        s2+=x[i+t]**2
        cov+=x[i]*x[i+t]

    m1 /= rng
    m2 /= rng
    s1 = (s1/rng-m1**2)**0.5
    s2 = (s2/rng-m2**2)**0.5
    cov /= rng

    return (cov-m1*m2)/(s1*s2)

m_ac = np.zeros(tmax)
s_ac = np.zeros(tmax)
m_ac_shuffled = np.zeros(tmax)
s_ac_shuffled = np.zeros(tmax)

for j in range(m):
    x = np.empty(nn)
    x[0] = 0.1
    noise = np.random.normal(0,np.sqrt(2),nn)
    for i in range(nn):
        x[i]= x[i-1]-y*x[i-1]*dt+np.sqrt(dt)*noise[i]

    w = x[::t]

    for t in range(tmax):
        ac = AC(w,t)
        m_ac[t] += ac
        s_ac[t] += ac**2

    x_shuffled = w.copy()

    for i in range(1,n):
        pos = random.randrange(n)
        temp = x_shuffled[i]
        x_shuffled[i] = x_shuffled[pos]
        x_shuffled[pos] = temp

    for t in range(tmax):
        ac = AC(x_shuffled,t)
        m_ac_shuffled[t]+=ac
        s_ac_shuffled[t]+=ac**2

for t in range(tmax):
    m_ac[t]/=m
    s_ac[t] = (s_ac[t]/m - m_ac[t]**2)**0.5
    m_ac_shuffled[t]/=m
    s_ac_shuffled[t] = (s_ac_shuffled[t]/m - m_ac_shuffled[t]**2)**0.5


x_plot = np.arange(tmax) * dt
plt.errorbar(x_plot, m_ac, yerr=s_ac, c='black', fmt='.', capsize=5, label='UL')
plt.errorbar(x_plot, m_ac_shuffled, yerr=s_ac_shuffled, c='red', fmt='.', capsize=5, label='UL - Shuffled')
plt.legend()
plt.yscale('log')
plt.show()
    