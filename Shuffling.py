import numpy as np
from numpy.random import randint

x = np.random.uniform(1,101,10) #float
n = len(x)
#NAIVE bias statistico
for i in range(n):
    j = randint(n) # [0,n-1]
    x[j], x[i] = x[i], x[j]

#DUSTERNFELD è fisher ottimizzato e veloce, shuffla in avanti invece che indietro senza array intermedi, permutazioni uniformi no ripetizione 
for i in range(n-1): # range(n-1,0,-1) e randint(0,i+1) per dusternfeld e fisher yates è quello accapo
    j = randint(i,n) # j = i +randint(n-i) [0,n-i]
    x[j], x[i] = x[i], x[j]

#SATTOLO j>1 no swap con se o precedenti, permutazioni cicliche no ripetizioni
for i in range(n-1):
    j = randint(i+1,n)  # j = i+1 +randint(n-i-1) [0,n-i-1]
    x[j], x[i] = x[i], x[j]

#BOOTSTRAP campiona con ripetizioni
x_b = np.zeros(n)
for i in range(n):
    j = randint(0,n)
    x_b[i] = x[j]