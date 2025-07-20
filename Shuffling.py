import random

# 0 
def naive(x):
    n = len(x)
    for i in range(n):
        j = random.randrange(n) #equivalente a random.randint(0,n-1) random tra 0 e n-1
        x[i], x[j] = x[j], x[i]
    return x

# i
def durstenfeld(x):
    n = len(x)
    for i in range(n-1):
        # j = i + random.randrange(n-i)
        j = random.randrange(i,n)
        x[i], x[j] = x[j], x[i]
    return x
#i+1
#The only difference with Durstenfeld is that j>i and not j>=i.
# Seleziona un indice successivo all'attuale (no swap con sé stesso)
def sattolo(x):
    n = len(x)
    for i in range(n-1):
        # j = i+1 + random.randrange(n-i-1)
        j = random.randrange(i+1,n)
        x[i], x[j] = x[j], x[i]
    return x


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

# ciclare sempre da 0 a n-1 eccetto per la versione naive
# naive numero a 0 a n-1, durstenfeld da 0 a n-i, sattolo da 0 a n-i-1,
