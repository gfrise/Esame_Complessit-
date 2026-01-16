from ast import Return
import numpy as np
import matplotlib.pyplot as plt

n = 10**2
x = np.zeros(n)
x[0] = 0
dt = 0.5
y = 0.1
mu = 0
std = 0.01

def ou():
    for i in range(1,n):
        x[i]=x[i-1]-y*dt*(x[i-1])+std*np.random.normal(0,1)
    return x

plt.plot(x)
plt.show()

n, l= 10**3, 10**2
means = np.empty(n)

for i in range(n):
    x = ou()
    means[i] = np.mean(x)

plt.hist(means)
plt.show()