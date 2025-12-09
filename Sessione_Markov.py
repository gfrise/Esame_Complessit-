import numpy as np
from numpy.linalg import eig

x = np.loadtxt("markov_chain.dat")
print(f'Catena di Markov:\n{x}')

states = np.array(x[:,1],dtype=int)
states = states - 1 #Così partono da 0
N = 3 # numero stati

#Stima matrice di transizione
trans = np.zeros((N,N), dtype=int)

for t in range(len(states)-1):
    i = states[t]
    j = states[t+1]
    trans[i,j]+=1

P = trans / trans.sum(axis=1, keepdims=True) #Normalizzazione
print(f'Matrice di transizione:\n{P}')

#Calcolo Autovalori e Autovettori
eigvals, eigvec = np.linalg.eig(P)
print(f'AUTOVALORI:\n{eigvals}\nAUTOVETTORI (colonne):\n{eigvec}')

sort_eigvals = np.sort(np.abs(eigvals))[::-1] #ordina modulo decrescente
print(f'AUTOVALORI ORDINATI:\n{sort_eigvals}')
print(f'Il secondo autovalore determina la convergenza alla stazionarietà')

# Determinazione stato stazionario
eigvalsT, eigvecT = np.linalg.eig(P.T) # Autovalori e autovettori della matrice trasposta, poichè pi_staz.T = P.T x pi_staz.T
stationarity = eigvecT[:, np.isclose(eigvalsT, 1)].real # Prendo la parte reale dell'autovettore associato ad 1
stationarity /= np.sum(stationarity) # Normalizzo l'autovettore associato ad 1
print(f'STAZIONARIETA\':\n{stationarity}')
print(f'L \'autovettore associato al primo autovalore (1) corrisponde alla distribuzione di probabilità in cui convergerà il processo stocastico\nSi nota che al passo t->inf gli stati non sono equiprobabili, poichè in la matrice di transizione è asimetrica, in particolare favorisce lo stato 1')

# Controllo sul vettore stazionarietà, deve valere anche pi_staz = pi_staz x P
print("Check pi_staz = pi_staz x P:", np.allclose(stationarity.T @ P, stationarity.T))
print("Check pi_staz.T = P.T x pi_staz.T:", np.allclose(P.T @ stationarity, stationarity))
