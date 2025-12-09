import numpy as np, matplotlib.pyplot as plt

n, l, m, sd = 10**3, 10**3, 3, 0.14
means = np.empty(n)

for i in range(n):
    x = np.random.normal(m,sd,l)
    means[i] = np.mean(x)

print("Media:",np.mean(means))
print("Sigma stimata:",np.std(means))
print("Sigma teorica:",sd/(l)**0.5)
plt.hist(means,bins=30)
plt.title("PDF Medie")
plt.show()
#Campiona con ripetizione (no permutazioni) da un campione non da tanti, riproduce più caratteristiche
original = np.random.normal(m,sd,l)

for i in range(n):
    x = np.random.choice(original,l,replace=True)
    # x = original[np.random.randint(0,l,l,dtype=int)]  [0,l-1] lungo l   
    means[i] = np.mean(x)

print("Medie:",np.mean(means))
print("Sigma stimata:", np.std(means))
print("Sigma teorica:", sd/(l)**0.5)
plt.hist(means,bins=30)
plt.title("PDF Medie con ripetizione")
plt.show()    