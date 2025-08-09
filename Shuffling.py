import numpy as np
from numpy.random import randint

np.random.seed(42)
x = np.random.uniform(1,101,10) #float
x = np.random.randint(1,101,10,dtype=int) #[1,101) lungo 10
x = np.arange(1,11)#ordinati interi
n = len(x)
#NAIVE mescola casualmente, introduce un bias statistico
for i in range(n):
    j = randint(n) # numero random tra 0 e n-1
    x[j], x[i] = x[i], x[j]

#DUSTERNFELD, ottimizza fisher yates, permutazioni uniformi, no ripetizione
for i in range(n-1): # range(n-1,0,-1) e randint(0,i+1) per dusternfeld e fisher yates è quello accapo
    j = randint(i,n) # = a j = i + random.randrange(n-i)
    x[j], x[i] = x[i], x[j]

#SATTOLO j>1 no swap con se stesso o precedenti, permutazioni cicliche, no ripetizioni
for i in range(n-1):
    j = randint(i+1,n)  # = a j = i+1 + random.randrange(n-i-1)
    x[j], x[i] = x[i], x[j]

#BOOTSTRAP campiona con ripetizioni
x_b = np.zeros(n)
for i in range(n):
    j = randint(0,n)
    x_b[i] = x[j]
    #oppure x_b = np.random.choice(x, size=n, replace= True)
print(x_b)

#differenza naive_bootstrap
## Primo metodo: shuffle (permutazione) – rimescola gli elementi di x senza ripetizioni.
# Secondo metodo: bootstrap sample – estrae elementi casuali da x con rimpiazzo (possono esserci ripetizioni).
# naive numero a 0 a n-1, durstenfeld da 0 a n-i, sattolo da 0 a n-i-1,

#dusternfeld ammette ripetizioni sattolo no, non può ricadere nello stesso posto 
# perchè c'è >= (fisher yates e dusternefel non c'è differenze, 
# durstenfeld è fisher yates ottimizzato perchè fisher yeates 
# ogni volta cencella il dato che shuflluato dal vettre originale)
