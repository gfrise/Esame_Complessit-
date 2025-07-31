import numpy as np
import matplotlib.pyplot as plt

# Compito 1: create_OU.c
# processo di Ornstein-Uhlenbeck a partire dalla sua equazione di Langevin. 
# Il codice deve anche prevedere la 
# 
# costruzione della pdf (area normalizzata ad 1) e funzione di autocorrelazione
# i parametri rilevanti della simulazione tramite file 
# Iterate il processo M = 100 volte e costruite l’istogramma e 
# l’autocorrelazione mediati su queste M iterazioni,
# mostrando la standard deviation come barra d’errore.

t, step, taum, m, y = 10**4, 100, 50, 100, 0.1
dt, n, means, sd = 1/step, t*step, np.zeros(taum), np.zeros(taum)

# def Ac(x):
#     n = len(x) - taum
#     m1, sd1, res = np.mean(x[:n]), np.std(x[:n]), np.zeros(taum)
#     for t in range(taum):
#         x2 = x[t:t+n]
#         m2, sd2, corr = np.mean(x2), np.std(x2), np.mean(x[:n]*x2) #x[j]*x[j+t]
#         res[t] = (corr - m1*m2) / (sd1 * sd2)
#     return res
# #Ornstein e Ulembeck ha h(x) = -gamma*x e g(x)=c, poniamo c=1
# for k in range(m):
#     x = np.zeros(n)
#     x[0] = 0.1
#     noise = np.random.normal(0, np.sqrt(2), n)
#     for i in range(1, n):
#         x[i] = x[i-1] - y*x[i-1]*dt + np.sqrt(dt)*noise[i]
#     ac = Ac(x[::step])
#     means += ac
#     sd += ac**2

# means /= m
# sd = np.sqrt(sd/m-means**2)
# plt.errorbar(np.arange(taum), means, yerr=sd, fmt='o', color='black')
# plt.yscale("log")
# plt.xlabel('Lag')
# plt.ylabel('Autocorrelation')
# plt.title('Ornstein-Uhlenbeck Autocorrelation')
# plt.show()

#   for(t=0;t<tmax;t++){
#      m2=0.;
#      sd2=0.;
#      corr=0.;
#      for(j=0; j<nR-tmax;j++){
# 	 m2=m2+X[j+t];
# 	 sd2=sd2+pow(X[j+t],2.);
# 	 corr=corr+X[j]*X[j+t];
#         }
# 	m2=m2/(double)(nR-tmax);
# 	sd2=(sd2/(double)(nR-tmax))-pow(m2,2.);
# 	sd2=pow(sd2,0.5);
# 	corr=corr/(double)(nR-tmax);
#         fprintf(fp3,"%d %lf \n", t, (corr-m1*m2)/(sd1*sd2)); 
#         }
#   fclose(fp3);  


x, noise = np.zeros((m, n)), np.random.normal(0, np.sqrt(2 * dt), (m, n)) #m paths paralleli
for i in range(1, n):
    x[:, i] = x[:, i-1] - y * x[:, i-1] * dt + noise[:, i]
x_coarse = x[:, ::step]  # shape (m, t)# --- Prendi solo i punti ogni 'step' ---
x_demean = x_coarse - np.mean(x_coarse, axis=1, keepdims=True)# --- Rimuovi la media per ogni traiettoria ---
acf = np.empty((m, taum))
for tau in range(taum):# --- Calcola autocorrelazione empirica fino a taum ---
    prod = x_demean[:, :-tau or None] * x_demean[:, tau:]
    acf[:, tau] = np.mean(prod, axis=1)
acf /= acf[:, [0]]# --- Normalizza per la varianza a lag 0 ---
mean_acf = np.mean(acf, axis=0)
std_acf = np.std(acf, axis=0, ddof=0)# --- Media e deviazione standard ---
plt.errorbar(np.arange(taum), mean_acf, yerr=std_acf, fmt='o', color='black')
plt.yscale('log')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Ornstein-Uhlenbeck Autocorrelation (NumPy Only)')
plt.show()

