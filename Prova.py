import numpy as np, matplotlib.pyplot as plt

def time(x,tau):
    x -= x.mean()
    return np.mean(x[:-tau]*x[tau:])/np.var(x)

def ens(x,tau):
    x -= x.mean(axis=1,keepdims=True)
    ac = np.mean(x[:,:-tau]*x[:,tau:],axis=1)/np.var(x,axis=1)
    return ac.mean()

def mix(x,tau):
    vars = np.var(x,axis=1,keepdims=True)
    x -= x.mean(axis=1,keepdims=True)
    ac = np.mean(x)/np.var(x)


# def timeavg(x,tau):
#     x = x - x.mean()
#     return np.mean(x[:-tau]*x[tau:])/np.var(x)

# def ensavg(x,tau):
#     x = x - x.mean(axis=1, keepdims=True)
#     ac = np.mean(x[:,:-tau]*x[:,tau:],axis=1)/np.var(x, axis=1)
#     return ac.mean(), ac.std()

# def mixavg(x,tau):
#     means = np.mean(x,axis=1,keepdims=True)
#     vars = np.var(x,axis=1,keepdims=True)
#     x = x - means
#     ac = np.mean(x[:,:-tau]*x[:,tau:]/vars)
#     return ac

def OU(T_tot, dt, m, y, sd):
    T = int(T_tot/dt)
    x = np.zeros(T)
    x[0] = 0.1
    for i in range(1,T):
        x[i] = x[i-1] + y*dt*(m - x[i-1]) + sd*np.random.normal(0,np.sqrt(2*dt))
    return x

def timeavg(x,tau):
    x = x - x.mean()
    return np.mean(x[:-tau]*x[tau:])/np.var(x)

def ensemble(N, T_tot, dt, m, y, sd):
    T = int(T_tot/dt)
    x = np.zeros((N,T))
    x[:,0] = 0.1

    for j in range(N):
        for i in range(1,T):
            x[j,i] = x[j,i-1] + y*dt*(m - x[j,i-1]) + sd*np.random.normal(0,np.sqrt(2*dt))

    return x

def ensavg(x,tau):
    x = x - x.mean(axis=1, keepdims=True)
    ac = np.mean(x[:,:-tau]*x[:,tau:], axis=1)/np.var(x,axis=1)
    return ac.mean()

x =OU(10**4,0.1,0,0.1,1)
t_ac = timeavg(x,30)

N = 1
while True:
    X = ensemble(N,10**4,0.1,0,0.1,1)
    ac_ens = ensavg(X,30)

    if np.abs(ac_ens - t_ac) < 0.1:
      break

    N = N+1

print(N)

