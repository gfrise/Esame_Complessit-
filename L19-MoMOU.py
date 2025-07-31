#  printf("\n Inizio di tutte le simulazioni: Avvio le iterazioni \n");
#   int iter;
#   double emme[nENS],emme2[nENS];
#   for(iter=0;iter<nENS;iter++){
  
#   double *Z1;
#   Z1=allocamem1(staz*nR*enne);
#   double x1,x2,gam;
#   double m=0;
#   double s=2*delt;
#   seed=19999999+rand()%100000001;

#   srand(seed);
#   for(i=0; i<staz*nR*enne;i++){
#          x1=(float)rand()/(float)RAND_MAX;
# 	  if(x1>0){
# 	     x2=(float)rand()/(float)RAND_MAX;
# 	     gam=sqrt(-2.*log(x1))*sin(2.*PIgreco*x2);
# 	     Z1[i]=m+(double) sqrt(s)*gam;
# 	  }
#     }
 
 
#   double *delXs,*Xs;
#   delXs=allocamem1(staz*nR*enne);
#   Xs=allocamem1(staz*nR*enne);

#   Xs[0]=(double)rand()/RAND_MAX;
#   Xs[0]=Xs[0]*2.-1.;
#   for(i=0; i<staz*nR*enne;i++){
#       delXs[i]=-gammaOU*Xs[i-1]*delt+Z1[i];
#       Xs[i]=Xs[i-1]+delXs[i];
#   }

#   double *X;
#   int p;
#   X=allocamem1(nR);
#   for(i=0; i<staz*nR*enne;i++){
#      p=(double) i/enne;
#      X[p]=Xs[i];
#   }

#   emme[iter]=mymedia(staz*nR,X);
  
#   free(delXs);
#   free(Z1);
#   free(Xs);
#   free(X);

#   }
#   printf("\n Fine di tutte le simulazioni: Chiudo le iterazioni \n");
  
#   double emmeav,emme2av,emme3av,emme4av,emme5av,emme6av;
#   emmeav=mymedia(nENS,emme);
#   emme2av=mymom2(nENS,emme);
#   emme3av=mymom3(nENS,emme);
#   emme4av=mymom4(nENS,emme);
#   emme5av=mymom5(nENS,emme);
#   emme6av=mymom6(nENS,emme);

#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Simulazione Ensemble-Average di:
  • Ornstein–Uhlenbeck
  • Processo multiscala di Risken
Calcolo di pdf (media±σ), autocorrelazione (media±σ) e momenti centrali ordine 1…K.
Tutti i parametri definiti internamente.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Parametri ───────────────────────────────────────────────────────
M, N, dt     = 100, 1000, 1e-2      # ensemble size, steps, timestep
σ            = 1.0                  # rumore gaussiano
bins, xlim   = 50, (-5,5)           # pdf
K            = 6                    # momenti fino all'ordine K

# ── Drift ───────────────────────────────────────────────────────────
def drift_OU(x, γ):        return -γ*x
def drift_RISK(x, α, β):   return -α*x/(1+β*x*x)

# ── Simulazione Euler–Maruyama ─────────────────────────────────────
def simulate(drift, params):
    X = np.zeros((M, N))
    sqrt_dt = np.sqrt(dt)
    for m in range(M):
        for i in range(1, N):
            X[m,i] = (X[m,i-1]
                      + drift(X[m,i-1], *params)*dt
                      + σ*np.random.randn()*sqrt_dt)
    return X

# ── Statistiche Ensemble ───────────────────────────────────────────
def stats(X):
    # PDF
    edges = np.linspace(*xlim, bins+1)
    h = np.array([np.histogram(traj, edges, density=True)[0] for traj in X])
    centers, hμ, hσ = (edges[:-1]+edges[1:])/2, h.mean(0), h.std(0)
    # Autocorr
    def acf(x):
        xc = x - x.mean()
        c  = np.correlate(xc, xc, 'full')[N-1:]
        return c/c[0]
    A = np.array([acf(x) for x in X])
    aμ, aσ = A.mean(0), A.std(0)
    # Momenti centrali su tutti i campioni
    allx = X.ravel()
    μ = allx.mean()
    cm = [np.mean((allx-μ)**k) for k in range(1, K+1)]
    return centers, hμ, hσ, aμ, aσ, cm

# ── Plot ────────────────────────────────────────────────────────────
def plot(centers, hμ, hσ, aμ, aσ, cm, title):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.errorbar(centers, hμ, yerr=hσ, fmt='o', elinewidth=1)
    plt.title(f'PDF {title}'); plt.xlabel('x'); plt.ylabel('densità')

    plt.subplot(1,2,2)
    τ = np.arange(len(aμ))*dt
    plt.errorbar(τ, aμ, yerr=aσ, fmt='-'); plt.yscale('log')
    plt.title(f'C(τ) {title}'); plt.xlabel('τ'); plt.ylabel('autocorr.')

    plt.tight_layout(); plt.show()

    print(f"\nMomenti centrali {title}:")
    for k,val in enumerate(cm,1):
        print(f"  ordine {k}: {val:.3e}")

# ── Esecuzione ─────────────────────────────────────────────────────
if __name__ == '__main__':
    # Ornstein–Uhlenbeck (γ)
    X_ou = simulate(drift_OU,    params=(0.7,))
    stats_ou = stats(X_ou)
    plot(*stats_ou, title='Ornstein–Uhlenbeck')

    # Risken multiscala (α,β)
    X_r  = simulate(drift_RISK,   params=(1.0,0.5))
    stats_r = stats(X_r)
    plot(*stats_r,  title='Risken multiscala')

# means = np.zeros(nENS)

# # Loop sull'ensemble
# total_steps = staz * nR * enne
# for iter_idx in range(nENS):
#     # Generazione rumore gaussiano con NumPy
#     Z1 = np.random.normal(loc=0.0, scale=np.sqrt(2 * delt), size=total_steps)

#     # Simulazione processo OU
#     Xs = np.zeros(total_steps)
#     Xs[0] = np.random.rand() * 2.0 - 1.0  # punto iniziale in [-1,1]
#     for i in range(1, total_steps):
#         dX = -gammaOU * Xs[i-1] * delt + Z1[i]
#         Xs[i] = Xs[i-1] + dX

#     # Sottocampionamento ogni 'enne' step
#     X = Xs[::enne][:nR]
#     means[iter_idx] = np.mean(X)

# # Calcolo dei momenti sull'ensemble delle medie
# momenti = [np.mean(means)] + [central_moment(means, k) for k in range(2,7)]

# e anche 
# means = np.empty(nENS)
# for k in range(nENS):
#     # rumore gaussiano
#     Z = np.random.normal(0, np.sqrt(2*delt), total_steps)
#     X = np.empty(total_steps)
#     X[0] = np.random.uniform(-1, 1)
#     for i in range(1, total_steps):
#         X[i] = X[i-1] + (-gamma * X[i-1] * delt + Z[i])
#     # sottocampionamento (qui è già nR*enne, quindi prendo tutto)
#     means[k] = X.reshape(nR, enne).mean(axis=1).mean()

# #--- Momenti sull'ensemble delle medie ---
# moments = [means.mean()] + [central_moment(means, m) for m in range(2,7)]

# #--- Scrittura ---