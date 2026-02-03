import numpy as np

T_tot, dt, y = 10**3, 0.1, 0.2
T = int(T_tot/dt)

def create_ensemble(N):
    x = np.zeros((N,T))
    x[:,0]

    for j in range(N):
        for i in range(1,T):
            x[j,i] = x[j,i-1] - y*dt*x[j,i-1] + np.random.normal(0,np.sqrt(2*dt))

    return x

def ensvag(x,tau):
    x = x - x.mean(axis=1,keepdims=True)
    N,T = x.shape
    ac = np.mean(x[:,:T-tau]*x[:,tau:],axis=1)/np.var(x,axis=1)
    return ac.mean()

def ac(taum):
    ac = np.zeros(taum+1)

    for tau in range(taum+1):
        ac[tau] = ensvag(create_ensemble(2),tau)

    return ac.mean()

print(ac(30))
