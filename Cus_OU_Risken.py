import numpy as np
import matplotlib.pyplot as plt
import random

N = 10**3
step = 100
dLag = 1/step
lagMax = 40
lag = np.arange(0, lagMax, dLag)
nn = N*step
m=100

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
    #print(corr)
    return (corr-m1*m2)/(sd1*sd2)

#Ornstein e Ulembeck ha h(x) = -gamma*x e g(x)=c, poniamo c=1
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

    x_ul_series = [x_ul[t] for t in range(1, nn, step)]
    x_risk_series = [x_risk[t] for t in range(1, nn, step)]
    
    for t in range(lagMax):
        ac = autocorrelation(x_ul_series, t)
        med_ul[t] += ac
        sd_ul[t] += ac**2

    for t in range(lagMax):
        ac = autocorrelation(x_risk_series, t)
        med_risk[t] += ac
        sd_risk[t] += ac**2

    #Shuffling
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
    # print(f'{t}: {med_ul[t]}\t{sd_ul[t]}\t{med_ul_shuffled[t]}\t{sd_ul_shuffled[t]}')

for t in range(lagMax):
    med_risk[t] /= m
    sd_risk[t] = sd_risk[t]/m - med_risk[t]**2
    sd_risk[t] = np.sqrt(sd_risk[t])
    med_risk_shuffled[t] /= m
    sd_risk_shuffled[t] = sd_risk_shuffled[t]/m - med_risk_shuffled[t]**2
    sd_risk_shuffled[t] = np.sqrt(sd_risk_shuffled[t])
    # print(f'{t}: {med_risk[t]}\t{sd_risk[t]}\t{med_risk_shuffled[t]}\t{sd_risk_shuffled[t]}')

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