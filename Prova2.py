import numpy as np, matplotlib.pyplot as plt

T, dt, N, m, sd, y, taum = 10**4, 0.1, 10**3, 0.17, 0.28, 0.1, 40
taus = np.arange(0,taum+1)

x = np.zeros((N,T))
x[:,0] = 0.42
for j in range(N):
    for i in range(1,T):
        x[j,i] = x[j,i-1]-y*dt*(x[j,i-1]-m)+sd*np.random.normal(0,dt**0.5)

y = np.zeros(T)
y = x[0,:]

def timeavg(x,taum):
    ac = np.zeros(taum+1)
    mean, var, T = np.mean(x), np.var(x), len(x)

    for tau in taus:
        prod = (x[:T-tau]-mean)*(x[tau:]-mean)
        ac[tau] = np.mean(prod)/var

    return ac

def mixavg(x,taum):
    ac = np.zeros((N,taum+1))

    for i in range(N):
        mean = np.mean(x[i,:])
        var = np.var(x[i,:])
        for tau in taus:
            prod = (x[i,:T-tau]-mean)*(x[i,tau:]-mean)
            ac[i,tau]=np.mean(prod)/var

    return np.mean(ac, axis=0), np.std(ac, axis=0)

def ensavg(x,taum):
    ac = np.zeros((N,taum+1))
    mean = np.mean(x)
    var = np.var(x)

    for tau in taus:
        prod = (x[:,:T-tau]-mean)*(x[:,tau:]-mean)
        ac[:,tau] = np.mean(prod)/var

    return np.mean(ac, axis=0), np.std(ac, axis=0)

plt.semilogy()
plt.plot(taus, np.exp(-y*dt*taus),label="Teorico")
plt.plot(taus, timeavg(y,taum), label="Time")
plt.errorbar(taus, mixavg(x,taum)[0], yerr=mixavg(x,taum)[1], label="Mixed")
plt.errorbar(taus, ensavg(x,taum)[0], yerr=ensavg(x,taum)[1], label="Ensemble")
plt.legend()
plt.grid(True)
plt.title("Merda")
plt.xlabel("lag")
plt.ylabel("AC")
plt.show()
