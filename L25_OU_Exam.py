import numpy as np, matplotlib.pyplot as plt
import random

n, t, tmax, m, y = 10**4, 100, 50, 100, 0.1
dt, nn = 1/t, n*t
lag = np.arange(0,tmax,dt)

def AC(x,t):
    m1, m2, s1, s2, cov = 0,0,0,0,0
    rng = n-tmax

    for i in range(rng):
        m1+=x[i]
        m2+=x[i+t]
        s1+=x[i]**2
        s2+=x[i+t]**2
        cov+=x[i]*x[i+t]
    
    m1/=rng
    m2/=rng
    cov/=rng
    s1 = (s1/rng - m1**2)**0.5
    s2 = (s2/rng - m2**2)**0.5

    return (cov-m1*m2)/(s1*s2)

ms, mshf, sds, sdshf = np.zeros(tmax), np.zeros(tmax), np.zeros(tmax), np.zeros(tmax)

for k in range(m):
    x = np.zeros(nn)
    x[0] = 0.1
    noise = np.random.normal(0,np.sqrt(2*dt),nn)

    for i in range(1,nn):
        x[i]=x[i-1]-y*dt*x[i-1]+noise[i]

    z = x[::t]
    w = z.copy()

    for i in range(1,n):
        pos=random.randrange(n)
        temp = w[i]
        w[i] = w[pos]
        w[pos] = temp

    for t in range(tmax):
        ac = AC(z,t)
        ms[t] += ac
        sds[t] += ac**2
        ac_shf = AC(w,t)
        mshf[t]+=ac_shf
        sdshf[t]+=ac_shf**2

for t in range(tmax):
    ms[t]/=m
    sds[t]/=sds[t]/m-ms[t]**2
    sds[t]=np.sqrt(sds[t])
    mshf[t]/=m
    sdshf[t]/=sdshf[t]/m-mshf[t]**2
    sdshf[t]=np.sqrt(sdshf[t])

x_plot = np.arange(tmax)
plt.figure()
plt.errorbar(x_plot, ms, yerr=sds, fmt='.', capsize=5, label='Autocorr. originale')
plt.errorbar(x_plot, mshf, yerr=sdshf, fmt='.', capsize=5, label='Autocorr. shufflata')
plt.yscale('log')
plt.xlabel('Lag')
plt.ylabel('Autocorrelazione')
plt.title('Andamento dell’autocorrelazione (y-log)')
plt.legend()
plt.tight_layout()
plt.show()
