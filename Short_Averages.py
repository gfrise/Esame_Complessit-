import numpy as np

def ensavg(x,tau):
    x = x - x.mean(axis=1, keepdims=True)
    T = len(x)
    ac = np.mean(x[:,:T-tau]*x[:,tau:],axis=1)/np.var(x,axis=1)
    return ac.mean()

def mixavg(x,tau):
    x = x - x.mean(axis=1,keepdims=True)
    ac = np.mean(x[:,:len(x)-tau]*x[:,tau:])/np.var(x,axis=1,keepdims=True)
    return ac


def timeavg(x,tau):
    x -= x.mean()
    return np.mean(x[:len(x)-tau]*x[tau:])/np.var(x)

def ensvag(x,tau):
    x = x - x.mean(axis=1,keepdims=True)
    ac = np.mean(x[:,:T-tau]*x[:,tau:],axis=1)/np.var(x,axis=1)
    return ac.mean()
