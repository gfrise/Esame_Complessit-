import numpy as np, matplotlib.pyplot as plt

n, t, tmax, m, y = 10**3, 100, 50, 100, 0.1
dt, nn = 1/t, n*t # t:=step
m_ou, s_ou = np.zeros(tmax), np.zeros(tmax) 


# def ensemble_autocorr(m, nn):
#     ac = np.zeros(nn)
#     for _ in range(m):
#         x = OU(nn)
#         ac += x[0] * x
#     return ac / m
#     # Calcolo autocorrelazione per questa traiettoria
#     # Calcolo autocorrelazione media sull'ensemble


### vedere se funziona
# def ensemble_avg(x,tmax): # x è array (M,T) di traiettorie per righe
#     data -= data.mean(axis=0)
#     col1 = x[:,0] # : -> prendi ogni riga, 0 -> col 1 ==> vettore lungo m delle serie al tempo 0
#     var1 = np.var(col1)
#     ac, ac_std = np.empty(tmax)
#     for t in range(tmax+1):
#         p = (col1*x[:,t])/var1 # corrisponde a fare for i in range(M):somma += X[i][0] * X[i][tau] return somma / M
#         ac.append(p.mean())
#         ac_std.append(p.std())
#     return ac, ac_std



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
def AC(x,t):
    m1,s1,m2,s2,corr,l = 0,0,0,0,0,n-tmax
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

for k in range(m):
    x = np.zeros(nn)
    x[0]=0.1
    for i in range(1,nn):
        x[i] = x[i-1] - y*dt*x[i-1] + np.random.normal(0,(2*dt)**0.5)
    x_ou = x[::t]
    for t in range(tmax):
        ac = AC(x_ou,t)
        m_ou[t] += ac
        s_ou[t] += ac**2

m_ou /= m
s_ou = np.sqrt(s_ou/m - m_ou**2)

x_plot = np.arange(tmax)
plt.errorbar(x_plot, m_ou, yerr=s_ou, c='black', fmt='.', capsize=5, label='UL')
plt.legend()
plt.yscale('log')
plt.show()