import numpy as np, matplotlib.pyplot as plt
from scipy.linalg import eig

P = np.array([[ 0.9, 0.1], 
              [0.1, 0.9]]) 

eigvals = np.linalg.eigvals(P)
eigvals = np.sort(np.abs(eigvals))[::-1]
rate = eigvals[1]
print(f'Il rate di convergenza è: {rate:.2}')

# Evoluzione della distribuzione di probabilitò da pi(0) a pi(t)
# Nel tempo l'evoluzione è pi(t) = pi(0)*P^t

pi0 = ([1.0, 0.0]) #distr iniziale casuale
# Evoluzione per T passi
T = 20
pi_t = np.zeros((T+1, len(pi0)))
pi_t[0] = pi0

for t in range(1, T+1):
    pi_t[t] = pi_t[t-1] @ P # Moltiplicazione tra matrici

plt.plot(pi_t[:,0], label='Stato 0')
plt.plot(pi_t[:,1], label='Stato 1')
plt.axhline(np.linalg.matrix_power(P, 1000)[0,0], color='gray', linestyle='--', label='Stazionario')
plt.xlabel('Tempo')
plt.ylabel('Probabilità')
plt.title('Evoluzione della distribuzione')
plt.grid(True)
plt.show()
# La funzione stazionaria soddisfa anche P.T pi.T = pi.T dove con .T indico la trasposta, ovviamente soddisfa anche l'uguaglianza senza .T
w, v = eig(P.T)
stationary = v[:, np.isclose(w, 1)].real 
print("Distribuzione stazionaria:", stationary/stationary.sum())