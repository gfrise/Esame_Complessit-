import numpy as np, matplotlib.pyplot as plt
import random

t = 100
dt = 1/t
n = 100
nn = t*n
y = 0.1
m = 50

for _ in range(m):
    x = np.empty(nn)
    x[0] = 0.3
    noise = np.random.normal(0,3,nn)
    for i in range(1,nn):
        x[i]=x[i-1]-y*dt*x[i-1]+np.sqrt(dt)*noise[i]
        ac[i]+=x[0]*x[k]
    