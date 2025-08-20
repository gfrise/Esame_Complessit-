import random, numpy as np, matplotlib.pyplot as plt

# Il codice deve anche prevedere la costruzione della pdf 
# (area normalizzata ad 1) e funzione di autocorrelazione
# Iterate il processo M = 100 volte e costruite l’istogramma della densità di probabilità stazionaria normalizzata a 1 e 
# la funzione di autocorrelazione mediati su queste M iterazioni,
# mostrando la standard deviation come barra d’errore.

N = 200
step = 100 #t
dLag = 1/step #dt
lagMax = 50 
lag = np.arange(0, lagMax, dLag) # interi ordinati start,stop,step
nn = N*step 
m=100

# Equivalenti
# def AC(x,t):
#     if t == 0: 
#         return 1.0
#     x1,x2=x[:-t],x[t:]
#     return ((x1 * x2).mean()-x1.mean()*x2.mean())/(x1.std()*x2.std())

# def Ac_equiv(x):
#     res = np.zeros(taum)
#     n = len(x) - taum 
#     x1 = x[:n]         
#     m1, sd1 = x1.mean(), x1.std()

#     for t in range(taum):
#         x2 = x[t:t+n]
#         m2, sd2 = x2.mean(), x2.std()
#         corr = (x1 * x2).mean() #x[j]*x[j+t]
#         res[t] = (corr - m1*m2) / (sd1 * sd2)
#     return res


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

def autocorrelation(x, t):
    m1=0
    sd1=0

    for j in range(N-lagMax):
        m1 = m1 + x[j]
        sd1 = sd1 + x[j]**2

    m1 = m1 / (N-lagMax)
    sd1 = (sd1/(N-lagMax))-m1**2
    sd1 = sd1**0.5
    
    m2=0
    sd2=0
    corr=0
    for j in range(N-lagMax):
        m2 = m2+x[j+t]
        sd2 = sd2 + x[j+t]**2
        corr = corr + x[j]*x[j+t]

    m2 /= (N-lagMax)
    sd2 = (sd2/(N-lagMax))-m2**2
    sd2 = sd2**0.5
    corr /= (N-lagMax)
    return (corr-m1*m2)/(sd1*sd2)

#Ornstein e Ulembeck ha h(x) = -gamma*x e g(x)=1
gamma1 = 0.1
gamma2 = 0.2
med_ul = np.zeros(lagMax)
sd_ul = np.zeros(lagMax)
med_ul_shuffled = np.zeros(lagMax)
sd_ul_shuffled = np.zeros(lagMax)

med_risk = np.zeros(lagMax)
sd_risk = np.zeros(lagMax)
med_risk_shuffled = np.zeros(lagMax)
sd_risk_shuffled = np.zeros(lagMax)

kappa = 0.5

for k in (range(m)):
    x_ul = np.zeros(nn)
    x_risk = np.zeros(nn)
    x_ul[0] = 0.1
    x_risk[0] = 0.1
    noise1 = np.random.normal(0, np.sqrt(2), nn)
    noise2 = np.random.normal(0, np.sqrt(2), nn)
    
    for i in range (1,nn):
        x_ul[i] = x_ul[i-1] - gamma1*x_ul[i-1]*dLag + np.sqrt(dLag)*noise1[i]
        if x_risk[i-1] < 0:
            x_risk[i] = x_risk[i-1] + kappa * dLag + np.sqrt(dLag) * noise2[i]
        else:
            x_risk[i] = x_risk[i-1] - kappa * dLag + np.sqrt(dLag) * noise2[i]

    # Moto browniano (processo di Wiener)
    # x[0] = 0.1
    # for i in range(1,N):
    # x[i] = x[i-1] + np.random.normal(0,dt**0.5)

    x_ul_series = [x_ul[t] for t in range(1, nn, step)]
    # x_ul_series = x_ul[::step]
    x_risk_series = [x_risk[t] for t in range(1, nn, step)]
    
    for t in range(lagMax):
        ac = autocorrelation(x_ul_series, t)
        med_ul[t] += ac
        sd_ul[t] += ac**2

    for t in range(lagMax):
        ac = autocorrelation(x_risk_series, t)
        med_risk[t] += ac
        sd_risk[t] += ac**2

    x_ul_shuffled = x_ul_series.copy()
    x_risk_shuffled = x_risk_series.copy()
    
    for i in range (1,N):
        pos = random.randrange(N)
        temp = x_ul_shuffled[i]
        x_ul_shuffled[i] = x_ul_shuffled[pos]
        x_ul_shuffled[pos] = temp

    for i in range (1,N):
        pos = random.randrange(N)
        temp = x_risk_shuffled[i]
        x_risk_shuffled[i] = x_risk_shuffled[pos]
        x_risk_shuffled[pos] = temp

    for t in range(lagMax):
        ac_shuffled = autocorrelation(x_ul_shuffled, t)
        med_ul_shuffled[t] += ac_shuffled
        sd_ul_shuffled[t] += ac_shuffled**2

    for t in range(lagMax):
        ac_shuffled = autocorrelation(x_risk_shuffled, t)
        med_risk_shuffled[t] += ac_shuffled
        sd_risk_shuffled[t] += ac_shuffled**2


for t in range(lagMax):
    med_ul[t] /= m
    sd_ul[t] = sd_ul[t]/m - med_ul[t]**2
    sd_ul[t] = np.sqrt(sd_ul[t])
    med_ul_shuffled[t] /= m
    sd_ul_shuffled[t] = sd_ul_shuffled[t]/m - med_ul_shuffled[t]**2
    sd_ul_shuffled[t] = np.sqrt(sd_ul_shuffled[t])

for t in range(lagMax):
    med_risk[t] /= m
    sd_risk[t] = sd_risk[t]/m - med_risk[t]**2
    sd_risk[t] = np.sqrt(sd_risk[t])
    med_risk_shuffled[t] /= m
    sd_risk_shuffled[t] = sd_risk_shuffled[t]/m - med_risk_shuffled[t]**2
    sd_risk_shuffled[t] = np.sqrt(sd_risk_shuffled[t])

x_plot = np.linspace(0,lagMax,lagMax)
plt.errorbar(x_plot, med_ul, yerr=sd_ul, c='black', fmt='.', capsize=5, label='UL')
plt.errorbar(x_plot, med_ul_shuffled, yerr=sd_ul_shuffled, c='red', fmt='.', capsize=5, label='UL - Shuffled')
plt.errorbar(x_plot, med_risk, yerr=sd_risk, c='blue', fmt='.', capsize=5, label='RISK')
plt.errorbar(x_plot, med_risk_shuffled, yerr=sd_risk_shuffled, c='green', fmt='.', capsize=5, label='RISK - Shuffled')
plt.legend()
plt.yscale('log')
plt.show()

x_plot = np.linspace(0,lagMax,lagMax)
plt.errorbar(x_plot, med_ul, yerr=sd_ul, c='black', fmt='.', capsize=5, label='UL')
plt.errorbar(x_plot, med_ul_shuffled, yerr=sd_ul_shuffled, c='red', fmt='.', capsize=5, label='UL - Shuffled')
plt.errorbar(x_plot, med_risk, yerr=sd_risk, c='blue', fmt='.', capsize=5, label='RISK')
plt.errorbar(x_plot, med_risk_shuffled, yerr=sd_risk_shuffled, c='green', fmt='.', capsize=5, label='RISK - Shuffled')
plt.legend()
plt.show()

x_plot = np.linspace(0,lagMax,lagMax)
plt.axhline(1/N**0.5, linewidth=2, linestyle='--')
plt.plot(x_plot, sd_ul, '.', c='black', label='UL')
plt.plot(x_plot, sd_ul_shuffled, '.', c='r', label='UL - Shuffled')
plt.plot(x_plot, sd_risk, '.', c='blue', label='RISK')
plt.plot(x_plot, sd_risk_shuffled, '.', c='green', label='RISK - Shuffled')
plt.legend()
plt.show()