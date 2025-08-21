import numpy as np, matplotlib.pyplot as plt
import random

# costruire pdf con area 1 e funzione di autocorrelazione
# Iterate M volte e costruite l’istogramma della densità di probabilità stazionaria normalizzata a 1 e 
# la funzione di autocorrelazione mediati su queste M iterazioni mostrando la standard deviation come barra d’errore. 

# col1 = x[:,0] # : -> prendi ogni riga, 0 -> col 1 ==> vettore lungo m delle serie al tempo 0
# for i in range(M):somma += X[i][0] * X[i][tau] return somma / M
def ensemble_autocorr_corrected(m, nn):
    ac = np.zeros(nn)
    ac2 = np.zeros(nn)
    for _ in range(m):
        x=np.random.normal(0,1)
        x_c = x - x.mean()
        var_x = x_c.var()
        for tau in range(nn):
            ac_tau = np.mean(x_c[:nn-tau]*x_c[tau:])
            ac[tau] += ac_tau/var_x
            ac2[tau] += (ac_tau/var_x)**2

    ac_mean = ac / m
    ac_std = np.sqrt(ac2/m - ac_mean**2)
    return ac_mean, ac_std

n, t, tmax, m, y, kappa = 1000, 100, 50, 100, 0.1, 0.5
dt, nn = 1/t, n*t # t:=step
m_ou, s_ou = np.zeros(tmax), np.zeros(tmax)
m_risk, s_risk = np.zeros(tmax), np.zeros(tmax)
m_ou_sh, s_ou_sh = np.zeros(tmax), np.zeros(tmax)
m_risk_sh, s_risk_sh = np.zeros(tmax), np.zeros(tmax)

def AC(x,t):
    m1,s1,m2,s2,corr,l = 0,0,0,0,0, len(x)-t #n-tmax
    for j in range(l):
        m1+=x[j]
        s1+=x[j]**2
        m2+=x[j+t]
        s2+=x[j+t]**2
        corr+=x[j]*x[j+t]
    m1 /= l
    m2 /= l
    s1 = (s1/l - m1**2)**0.5
    s2 = (s2/l - m2**2)**0.5
    return (corr/l - m1*m2)/(s1*s2)

#h(x) = -gamma*x e g(x)=1
for k in range(m):
    x_ou = np.zeros(nn)
    x_risk = np.zeros(nn)
    x_ou[0] = x_risk[0] = 0.1

    for i in range(1, nn):
        x_ou[i] = x_ou[i-1] - y*dt*x_ou[i-1] + np.random.normal(0,np.sqrt(2*dt))
        drift = kappa*dt if x_risk[i-1]<0 else -kappa*dt
        x_risk[i] = x_risk[i-1] + drift + np.random.normal(0,np.sqrt(2*dt))

    x_ou_s, x_risk_s = x_ou[::t], x_risk[::t]
    x_ou_shuff, x_risk_shuff = x_ou_s.copy(), x_risk_s.copy()

    for i in range(1,n): 
        j = random.randrange(n); x_ou_shuff[i],x_ou_shuff[j] = x_ou_shuff[j],x_ou_shuff[i]
        x_risk_shuff[i],x_risk_shuff[j] = x_risk_shuff[j],x_risk_shuff[i]

    for tau in range(tmax):
        ac = AC(x_ou_s,tau); m_ou[tau]+=ac; s_ou[tau]+=ac**2
        ac = AC(x_risk_s,tau); m_risk[tau]+=ac; s_risk[tau]+=ac**2
        ac = AC(x_ou_shuff,tau); m_ou_sh[tau]+=ac; s_ou_sh[tau]+=ac**2
        ac = AC(x_risk_shuff,tau); m_risk_sh[tau]+=ac; s_risk_sh[tau]+=ac**2

def finalize(mv,sv): return mv/m, np.sqrt(sv/m - (mv/m)**2) 
m_ou,s_ou = finalize(m_ou,s_ou)
m_risk,s_risk = finalize(m_risk,s_risk)
m_ou_sh,s_ou_sh = finalize(m_ou_sh,s_ou_sh)
m_risk_sh,s_risk_sh = finalize(m_risk_sh,s_risk_sh)

x_plot = np.arange(tmax)
plt.errorbar(x_plot,m_ou,yerr=s_ou,fmt='.',c='black',capsize=5,label='OU')
plt.errorbar(x_plot,m_ou_sh,yerr=s_ou_sh,fmt='.',c='red',capsize=5,label='OU - Shuffled')
plt.errorbar(x_plot,m_risk,yerr=s_risk,fmt='.',c='blue',capsize=5,label='RISK')
plt.errorbar(x_plot,m_risk_sh,yerr=s_risk_sh,fmt='.',c='green',capsize=5,label='RISK - Shuffled')
plt.plot(x_plot,np.exp(-y*x_plot),'k--',label='OU analitico')
plt.axhline(1/np.sqrt(n),ls='--',c='gray',label='1/sqrt(N)')
plt.legend()
plt.yscale('log')
plt.xlabel("Lag")
plt.ylabel("Autocorrelazione")
plt.show()