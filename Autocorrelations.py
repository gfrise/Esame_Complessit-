import numpy as np

def time_avg(x,tau):
    n = len(x)-tau
    acc = 0
    for i in range(n):
        acc+=x[i]*x[i+tau]
    return acc/n

def ensemble_avg
    

   