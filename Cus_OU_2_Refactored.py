import numpy as np
import matplotlib.pyplot as plt

# Simulazione Ensemble-Average di:
#   • Ornstein–Uhlenbeck
#   • Processo multiscala di Risken
# Calcolo di pdf (media±σ), autocorrelazione (media±σ) e momenti centrali ordine 1…K.

# def drift_OU(x, γ):        return -γ*x
# def drift_RISK(x, α, β):   return -α*x/(1+β*x*x)
h = lambda x : -2.*x
h1 = lambda x : -2.
g = lambda x : 1.

t, step, taum, m, y = 300, 100, 50, 100, 0.1
dt, n, means, sd = 1/step, t*step, np.zeros(taum), np.zeros(taum)

def Ac(x):
    n = len(x) - taum
    m1, sd1, res = np.mean(x[:n]), np.std(x[:n]), np.zeros(taum)
    for t in range(taum):
        x2 = x[t:t+n]
        m2, sd2, corr = np.mean(x2), np.std(x2), np.mean(x[:n]*x2) #x[j]*x[j+t]
        res[t] = (corr - m1*m2) / (sd1 * sd2)
    return res

def OU2():
    x = np.zeros(n)
    x[0] = 0
    dw = np.random.normal(0,np.sqrt(2*dt),n)

    for i in range(1,n):
        xi = x[i-1]
        dx = h(xi)*dt+g(xi)*dw[i-1] + 0.5*h1(xi)*g(xi)*(dw[i-1]**2-2*dt)
        x[i]=xi+dx
    return x

def OU2(n,sigma,dt,y):
    x = np.empty(n)
    x[0] = 0.1
    
    g1, g2 = np.random.normal(0,1,n-1), np.random.normal(0,1,n-1)
    z1 = sigma*np.sqrt(dt)*g1
    z2 = sigma*(g1/2 + g2/(2*np.sqrt(3)))*(dt**1.5) ## senza sigm probabilmente
    ydt = y*dt

    for i in range(1,n):
        x[i] = x[i-1]*(1-ydt+0.5*ydt**2) + z1[i-1] - y*z2[i-1] 
    
    return x

for k in range(m):
    x = np.zeros(n)
    x[0] = 0.1
    for i in range(1, n):
        x[i] = x[i-1] - y*x[i-1]*dt + np.random.normal(0,(2*dt)**0.5)
        #x[i+1]=x[i]+h(x[i])*dt+g(x[i])*dw[i-1]     h(x)=-yx
    ac = Ac(x[::step])
    means += ac
   # means[i] = np.mean(x[::step]) ?
    sd += ac**2

means /= m
sd = np.sqrt(sd/m-means**2)
plt.errorbar(np.arange(taum), means, yerr=sd, fmt='o', color='black')
plt.yscale("log")
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Ornstein-Uhlenbeck Autocorrelation')
plt.show()

