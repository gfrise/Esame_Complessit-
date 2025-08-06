import numpy as np
import random

# Shuffling
# dusternfeld ammette ripetizioni sattolo no, non può ricadere nello stesso posto 
# perchè c'è >= (fisher yates e dusternefel non c'è differenze, 
# durstenfeld è fisher yates ottimizzato perchè fisher yeates 
# ogni volta cencella il dato che shuflluato dal vettre originale)

np.random.seed(42)
#float 
x = np.random.uniform(1,101,10)
# interi
x = np.random.randint(1,101,10,dtype=int) #[1,101) lungo 10
#interi ordinati
x = np.arange(1,11)

#0
def naive(x):
    n = len(x)
    for i in range(n):
        j = random.randrange(n) # numero random tra 0 e n-1, uguale a random.randint(0,n-1)
        x[i], x[j] = x[j], x[i]
    return x

#i 
def durstenfeld(x):
    n = len(x)
    for i in range(n-1):
        j = random.randrange(i,n) # = a j = i + random.randrange(n-i)
        x[i], x[j] = x[j], x[i]
    return x

#i+1
#The only difference with Durstenfeld is that j>i and not j>=i.
# Seleziona un indice successivo all'attuale (no swap con sé stesso)
def sattolo(x):
    n = len(x)
    for i in range(n-1):
        j = random.randrange(i+1,n) # = a j = i+1 + random.randrange(n-i-1)
        x[i], x[j] = x[j], x[i]
    return x

# CUSENZA 

#Bootstrap: é uno shuffling senza rimpiazzi
np.random.seed(42)
n = len(x)
x_b = np.zeros(n)
for i in range(n):
    j = np.random.randint(0,n)
    x_b[i] = x[j]
    #oppure x_b = np.random.choice(x, size=n, replace= True)
print(x_b)

#Durstenfeld: E' un ottimizzazione del metodo Fisher-Yates. 
# Parte dal primo elemento e va verso l'ultimo.
# Ad ogni iterazione scambia con un elemento compreso tra i ed N
np.random.seed(42)
n = len(x)
x_d = x.copy()
for i in range(0,n-1):
    j = np.random.randint(i,n)
    temp = x_d[i]
    x_d[i] = x_d[j]
    x_d[j] = temp
print(x_d)

#Sattolo: Come Durstenfeld ma non consente ripetizioni
np.random.seed(42)
n = len(x)
x_s = x.copy()
for i in range(n-1):
    j = np.random.randint(i+1,n)
    temp = x_s[i]
    x_s[i] = x_s[j]
    x_s[j] = temp
print(x_s)

#Naive: mescola casualmente, introduce un bias statistico
np.random.seed(42)
n = len(x)
x_n = x.copy()
for i in range(n):
    j = np.random.randint(0,n):
    temp = x_n[i]
    x_n[i] = x_n[j]
    x_n[j] = temp
print(x_n)

#differenza naive_bootstrap
## Primo metodo: shuffle (permutazione) – rimescola gli elementi di x senza ripetizioni.
# Secondo metodo: bootstrap sample – estrae elementi casuali da x con rimpiazzo (possono esserci ripetizioni).
# ciclare sempre da 0 a n-1 eccetto per la versione naive
# naive numero a 0 a n-1, durstenfeld da 0 a n-i, sattolo da 0 a n-i-1,

## UNDECIDED

def fisher_yates_original(arr):
    """Algoritmo Fisher-Yates originale (1938)"""
    n = len(arr)
    for i in range(n-1, 0, -1):  # Dall'ultimo elemento al primo
        j = random.randint(0, i)  # Seleziona casualmente da 0 a i ma nelle slide del prof [0,n-i-1]
        arr[i], arr[j] = arr[j], arr[i]
    return arr

def fisher_primo(x):
    n = len(x)
    for i in range(n):
        j = random.randrange(n-i)
        x[i],x[j] = x[j], x[i]
    return x

def fisher_yates_urn(arr):
    for step in range(len(arr)):
        # scegli indice random nell'intervallo dei rimanenti
        pick = random.randint(0, len(pool) - 1)
        # aggiungi l'elemento estratto al risultato
        result.append(pool.pop(pick))

    return result