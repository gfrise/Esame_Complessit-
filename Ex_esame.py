import numpy as np, matplotlib.pyplot as plt
import random

t, n, y, m, tmax = 100, 10**3, 0.1, 100, 50
nn, dt = n*t, 1/t
m_ac, sd_ac, m_acs, sd_acs = np.zeros(tmax), np.zeros(tmax), np.zeros(tmax), np.zeros(tmax),

def AC(x,t):
    m1,m2,s1,s2,cov = 0,0,0,0,0
    rng = len(x)-tmax

    for i in range(rng):
        m1+=x[i]
        m2+=x[i+t]
        s1+=x[i]**2
        s2+=x[i+t]**2
        cov+=x[i]*x[i+t]

    m1/=rng
    m2/=rng
    s1 = (s1/rng - m1**2)**0.5
    s2 = (s2/rng - m2**2)**0.5

    return (cov/rng -m1*m2)/(s1*s2)

for k in range(m):
    x = np.zeros(nn)
    x[0] = 0.1
    noise = np.random.normal(0,np.sqrt(2*dt),nn)

    for i in range(1,nn):
        x[i]=x[i-1]*(1-y*dt)+noise[i-1]

    x_ = x[::t]

    _x = np.random.choice(x_,len(x_),replace=True)

    for t in range(tmax):
        ac = AC(x_,t)
        m_ac[t] += ac
        sd_ac[t] += ac**2
        acc = AC(_x,t)
        m_ac[t] += acc
        sd_acs[t] += acc**2

for t in range(tmax):
    m_ac[t] /= m
    sd_ac[t] = (sd_ac[t]/m - m_ac[t]**2)**0.5
    m_acs[t] /= m
    sd_acs[t] = (sd_acs[t]/m - m_acs[t]**2)**0.5

x_plot = np.arange(tmax) * dt
plt.errorbar(x_plot, m_ac, yerr=sd_ac, c='black', fmt='.', capsize=5, label='UL')
plt.errorbar(x_plot, m_acs, yerr=sd_acs, c='red', fmt='.', capsize=5, label='UL - Shuffled')
plt.legend()
plt.yscale('log')
plt.show()
    