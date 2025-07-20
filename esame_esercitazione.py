import numpy as np, matplotlib.pyplot as plt

y, dt, t, n, nbin = 0.1, 0.1, 10**5, 100, 1000

x = np.zeros(t)
x[0] = 0.1
for i in range(1,t):
    x[i] = x[i-1]-y*dt*x[i-1]+np.random.normal(0,np.sqrt(2*dt))

bins = np.linspace(-5,5,nbin+1)
pdfs = np.zeros((n,nbin))

for i in range(n):
    sample = x[np.random.randint(0, t, size=t)]
    pdfs[i], _ = np.histogram(sample, bins=bins, density=True)

centers = (bins[1:]+bins[:-1])/2
mean_pdf = pdfs.mean(axis=0)
std_pdf = pdfs.std(axis=0)

plt.hist(x,bins=bins,density=True,alpha=0.5,label="PDF originale")
plt.errorbar(centers,mean_pdf,yerr=std_pdf,fmt='-',c ='r',label="Media +-1 sigma")
plt.show()