import numpy as np, matplotlib.pyplot as plt
# from numba import njit

T, dt, max_tau, N = 10**5, 1.0, 30, 100 #dt=1/step
taus = np.arange(0,max_tau+1)
mu, sigma, gamma = 0.0, 1.0, 0.1

x = np.zeros(T)
x[0] = 0.1
for t in range(1, T):
    noise = np.random.normal(0, np.sqrt(dt))
    x[t] = x[t-1] - gamma*(x[t-1]-mu)*dt + sigma*noise

# @njit
def ac_time_average(x, max_tau):
    T, mean, var = len(x), np.mean(x), np.var(x)
    ac = np.zeros(max_tau+1)
    for tau in range(max_tau+1):
        prod = (x[:T-tau] - mean) * (x[tau:] - mean)
        ac[tau] = np.mean(prod) / var
    return ac

ac_time = ac_time_average(x, max_tau)
ac_theoretical = np.exp(-gamma * taus * dt)

plt.semilogy()
plt.plot(taus, ac_time, 'o', ms=3, label="Time average AC")
plt.plot(taus, ac_theoretical, '--', label="Theoretical AC")
plt.xlabel("Lag τ")
plt.ylabel("Autocorrelazione")
plt.title("OU autocorrelazione vs teoria")
plt.legend()
plt.grid(True)
plt.show()

#media delle autocorrelazioni time-average per ogni traiettoria
def ac_mixed_average(X,max_tau):
    N,T = X.shape
    ac = np.zeros((N,max_tau+1))

    #ac time per ogni traiettoria
    for i in range(N):
        mean, var = np.mean(X[i,:]), np.var(X[i,:])

        for tau in taus:
            prod = (X[i,:T-tau]-mean)*(X[i,tau:]-mean)
            ac[i,tau] = np.mean(prod)/var

    ac_mix = np.mean(ac,axis=0)
    d_ac_mix = np.std(ac, axis=0)

    return ac_mix, d_ac_mix

X = np.zeros((N,T))
X[:,0] = 0.1
for i in range(N):
    for t in range(1,T):
        noise = np.random.normal(0, np.sqrt(dt))
        X[i,t] = X[i,t-1] - gamma*(X[i,t-1]-mu)*dt + sigma*noise
ac_mix, d_ac_mix = ac_mixed_average(X, max_tau)

#AC tra t e t+tau di tutte le traiettorie e poi si fa la media. si centrano tutte le traiettorie 
def ac_ensemble_average(X, max_tau):
    N, T = X.shape
    ac, d_ac = np.zeros(max_tau+1), np.zeros(max_tau+1)

    means = np.mean(X, axis=1, keepdims=True) # Media lungo ogni riga
    varsx = np.var(X, axis=1, keepdims=True) # Varianza lungo ogni riga
    Xc = X-means # Centra ogni riga i una volta per tutti

    for tau in range(max_tau+1):
        numeratori = np.sum(Xc[:, :T-tau]*Xc[:,tau:], axis=1) # Somma di tutte le correlazioni t - t+tau fissato e traiettoria i variabile
        denominatori = (T-tau)*varsx.flatten() # Quel T-tau serve per poi con il numeratore fanno la media della corr fra t - t+tau per le traiettorie
        ac_tau = numeratori/denominatori # Ha shape N, è la ac per ogni traiettoria
        d_ac[tau] = np.std(ac_tau)
        ac[tau] = np.mean(ac_tau) # Media sulle ac di ogni traiettoria

    return ac, d_ac

ac_ens, d_ac_ens = ac_ensemble_average(X,max_tau)

plt.semilogy()
plt.errorbar(taus, ac_ens, d_ac_ens, label="Ensemble ACF") #ac_mix,d_ac_mix
plt.xlabel("Lag τ")
plt.ylabel("Autocorrelazione")
plt.grid()
plt.legend()
plt.show()

# Milstein 2° anche rumore moltiplicativo
def milstein(n, x0, dt, h, g, g1):
    x = np.empty(n); x[0] = x0
    dW = np.random.normal(0, np.sqrt(dt), n-1)
    for i in range(1, n):
        xi = x[i-1]; d = dW[i-1]
        x[i] = xi + h(xi)*dt + g(xi)*d + 0.5*g(xi)*g1(xi)*(d*d - dt)
    return x

# OU 2° Kloeden-Platen per OU additivo
def OU2_second(n, sigma=np.sqrt(2.0), dt=1e-3, y=1.0, x0=0.1, seed1=None, seed2=None):
    x = np.empty(n); x[0] = x0
    rng1 = np.random.default_rng(seed1); rng2 = np.random.default_rng(seed2); 
    g1 = rng1.normal(0,1,n-1); g2 = rng2.normal(0,1,n-1); 
    Y1 = sigma * g1; Y2 = sigma * g2; 
    z1 = Y1 * np.sqrt(dt)
    z2 = (Y2/(2.0*np.sqrt(3.0)) + Y1/2.0) * dt**1.5
    ydt = y * dt
    for i in range(1, n):
        x[i] = x[i-1] * (1.0 - ydt + 0.5*(ydt**2)) + z1[i-1] - y * z2[i-1]
    return x

# Moto browniano (processo di Wiener)
def Wiener():
    x[0] = 0.1
    for i in range(1,N):
        x[i] = x[i-1] + np.random.normal(0,dt**0.5) 
