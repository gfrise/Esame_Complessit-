import numpy as np
import matplotlib.pyplot as plt

T, N, dt, m, s, y, taum = 10**5, 10**2, 0.1, 0.2, 0.3, 0.01, 40

x = np.zeros((N,T))
x[:,0] = 0.42
for i in range(N):
    for t in range(1,T):
        x[i,t]=x[i,t-1]-y*dt*(x[i,t-1]-m)+s*np.random.normal(0,np.sqrt(dt))

def acmix(x,taum):
    N,T = x.shape
    ac = np.zeros((N,taum+1))

    for i in range(N):
        mean = np.mean(x[i,:])
        var = np.var(x[i,:])

        for tau in range(taum+1):
            prod = (x[i,:T-tau]-mean)*(x[i,tau:]-mean)
            ac[i,tau] = np.mean(prod)/var
    
    return ac.mean(axis=0), ac.std(axis=0)

taus = np.arange(0,taum+1)
teo = np.exp(-y*dt*taus)
acm, acs = acmix(x,taum)

plt.semilogy()
plt.errorbar(taus, acm, acs, label="reale")
plt.plot(taus, teo, '--', label="teorico")
plt.xlabel("lag")
plt.ylabel("ac")
plt.title("ac vs lag")
plt.legend()
plt.grid(True)
plt.show()