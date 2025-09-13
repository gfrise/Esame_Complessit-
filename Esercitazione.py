import numpy as np, matplotlib.pyplot as plt

## bootstrap 
l,n = 10**3, 10**4
means = np.empty(n)
for i in range(1,n):
    x = np.random.normal(0,30,l)
    means[i]=np.mean(x)
print(means.mean())
print(30/l**0.5)
print(means.std())
plt.hist(means,bins=40)
plt.title("pdf medie")
plt.show()

##bootstrap da un campione
sample = np.random.normal(0,3,l)
for i in range(n):
    x = np.random.choice(sample,l,replace=True)
    means[i]=np.mean(x)
print(means.mean())
print(3/l**0.5)
print(means.std())
plt.hist(means,bins=40)
plt.title("Pdf medie")
plt.show()

### shuffle naive
z = np.random.uniform(0,101,20)
N = len(z)
for i in range(N):
    j = np.random.randint(N)
    z[i],z[j]=z[j],z[i]

### shuffle durstenfeld
for i in range(N-1):
    j = np.random.randint(i,N)
    z[i],z[j]=z[j],z[i]

## shuffle sattolo
for i in range(N-1):
    j = np.random.randint(i+1,N)
    x[i],x[j]=x[j],x[i]

### bootstrap replaement
x_b = np.zeros(len(z))




