import numpy as np, matplotlib.pyplot as plt
from scipy.linalg import eig

P = np.array([[0.9,0.1],
              [0.1,0.9]])

eigvals = np.linalg.eigvals(P)
eigvals = np.sort(np.abs(eigvals))[::-1]
print(eigvals[1])

pi0 = [1.0, 0.0]
T = 20
pi = np.zeros((T+1, len(pi0)))
pi[0] = pi0

for t in range(1,T+1):
    pi[t] = pi[t-1]@P

print(pi[T])

w,v = eig(P.T)
stationary = v[:, np.isclose(w, 1)]
stationary = stationary[:,0].real
stationary /= stationary.sum()

print(stationary)
