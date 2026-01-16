import numpy as np
import matplotlib.pyplot as plt

T, dt, N, taum = 10**2, 0.5, 2, 30
mu, std, y = 0.08, 0.17, 0.03

x = np.zeros(T)
x[0] = 0.1

for i in range(1,T):
    x[i]=x[i-1]-y*dt*x[i-1]+std*np.random.normal(0,np.sqrt(dt))

def ac_time(x,taum):
    mean = np.mean(x)
    ac = np.zeros(taum+1)
    for tau in range(taum+1):
        prod = (x[:len(x)-tau]-mean)*(x[tau:]-mean)
        ac[tau]=np.mean(prod)/np.var(x)
    return ac

print(ac_time(x,taum))


w =