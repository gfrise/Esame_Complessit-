import numpy as np
from numpy.linalg import eig

x = np.loadtxt("markov_chain.dat")
stati = np.array(x[:,1],dtype=int) - 1
n = 3
trans = np.zeros((n,n))

for t in range(len(stati)-1):
  i = stati[t]
  j = stati[t+1]
  trans[i,j]+=1

P = trans / trans.sum(axis=1, keepdims=True)
l, w = eig(P)

sort_l = np.sort(np.abs(l))[::-1]

lt, wt = eig(P.T)
statio = wt[:,np.isclose(lt,1)].real
statio /= np.sum(statio)

print(np.allclose(statio.T@P, statio.T))
print(np.allclose(P.T@statio, statio))